"""나스닥 일일 방향 예측기 (AND/OR 규칙 + 가점제)

사용법
  pip install yfinance pandas
  python predictor.py            # 오늘 신호 + 백테스트
  python predictor.py --demo     # 가짜 데이터로 동작 확인
  python predictor.py --html out.html

예측 대상: 다음 거래일 나스닥100(^NDX) 종가 대비 등락.
모든 지표는 당일 종가(t)까지만 사용 → t+1 예측 (미래참조 없음).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent

TICKERS = {
    "NDX": "^NDX", "NQ": "NQ=F", "SOX": "^SOX", "NVDA": "NVDA",
    "TNX": "^TNX", "VIX": "^VIX", "DXY": "DX-Y.NYB", "BTC": "BTC-USD",
    "HYG": "HYG", "IEF": "IEF",
}


# ---------------- 데이터 ----------------
def load_prices(years=10):
    import yfinance as yf
    raw = yf.download(list(TICKERS.values()), period=f"{years}y",
                      interval="1d", auto_adjust=True, progress=False)["Close"]
    inv = {v: k for k, v in TICKERS.items()}
    px = raw.rename(columns=inv)
    px = px[px["NDX"].notna()].ffill()          # 나스닥 거래일 기준 정렬
    return px.dropna()


def demo_prices(n=2500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-01", periods=n)
    common = rng.normal(0, 0.01, n)
    px = {}
    for k in TICKERS:
        r = 0.6 * common + rng.normal(0, 0.01, n)
        px[k] = 100 * np.exp(np.cumsum(r))
    px = pd.DataFrame(px, index=idx)
    # 데모용 선행관계 심기: 오늘 SOX 수익률이 내일 NDX에 일부 반영
    sox = px["SOX"].pct_change().fillna(0).values
    ndx_r = 0.3 * np.roll(sox, 1) + rng.normal(0, 0.01, n)
    px["NDX"] = 100 * np.exp(np.cumsum(ndx_r))
    px["TNX"] = 4 + np.cumsum(rng.normal(0, 0.04, n))
    px["VIX"] = np.clip(18 + np.cumsum(rng.normal(0, 0.5, n)) * 0.1, 10, 60)
    return px


# ---------------- 지표 ----------------
def features(px):
    f = pd.DataFrame(index=px.index)
    for k in px.columns:
        if k in ("TNX", "VIX"):
            f[f"{k}"] = px[k]
            f[f"{k}_chg1"] = px[k].diff()                # 수준 변화(포인트)
            f[f"{k}_pct1"] = px[k].pct_change()
        else:
            f[f"{k}_ret1"] = px[k].pct_change()
            f[f"{k}_ret5"] = px[k].pct_change(5)
        f[f"{k}_ma20"] = px[k] / px[k].rolling(20).mean() - 1  # 20일선 이격
    f["HY_ret1"] = (px["HYG"] / px["IEF"]).pct_change()       # 신용위험선호
    f["HY_ma20"] = (px["HYG"] / px["IEF"]) / (px["HYG"] / px["IEF"]).rolling(20).mean() - 1
    f["TARGET"] = px["NDX"].pct_change().shift(-1)            # 내일 수익률
    return f


# ---------------- 규칙 엔진 ----------------
OPS = {">": np.greater, "<": np.less, ">=": np.greater_equal, "<=": np.less_equal}


def evaluate(cond, f):
    """cond: [지표, 연산자, 값] | {"all":[...]} | {"any":[...]} | {"not":cond}"""
    if isinstance(cond, list):
        col, op, val = cond
        return pd.Series(OPS[op](f[col], val), index=f.index) & f[col].notna()
    if "all" in cond:
        out = pd.Series(True, index=f.index)
        for c in cond["all"]:
            out &= evaluate(c, f)
        return out
    if "any" in cond:
        out = pd.Series(False, index=f.index)
        for c in cond["any"]:
            out |= evaluate(c, f)
        return out
    if "not" in cond:
        return ~evaluate(cond["not"], f)
    raise ValueError(f"잘못된 조건: {cond}")


def score(rules, f):
    hits = pd.DataFrame({r["name"]: evaluate(r["when"], f) for r in rules})
    pts = pd.Series({r["name"]: r["points"] for r in rules})
    return hits, (hits * pts).sum(axis=1)


def signal(s, th):
    return np.where(s >= th, "상승", np.where(s <= -th, "하락", "중립"))


# ---------------- 백테스트 ----------------
def rule_report(rules, hits, f):
    y = f["TARGET"]
    base_up = (y > 0).mean()
    rows = []
    for r in rules:
        m = hits[r["name"]] & y.notna()
        n = int(m.sum())
        if n == 0:
            rows.append((r["name"], r["points"], 0, np.nan, np.nan, "데이터없음"))
            continue
        up = (y[m] > 0).mean()
        avg = y[m].mean() * 100
        right = (up > base_up) == (r["points"] > 0)
        edge = abs(up - base_up)
        verdict = "유효" if right and edge >= 0.03 and n >= 30 else ("약함" if right else "반대")
        rows.append((r["name"], r["points"], n, up * 100, avg, verdict))
    df = pd.DataFrame(rows, columns=["규칙", "점수", "발생일", "다음날상승%", "평균수익%", "판정"])
    return df, base_up


def score_report(s, f, th):
    d = pd.DataFrame({"s": s, "y": f["TARGET"]}).dropna()
    d["sig"] = signal(d["s"], th)
    by = d.groupby("s")["y"].agg(일수="count", 상승확률=lambda x: (x > 0).mean() * 100,
                                  평균수익=lambda x: x.mean() * 100)
    act = d[d.sig != "중립"]
    acc = ((act.sig == "상승") == (act.y > 0)).mean() * 100 if len(act) else np.nan
    # 신호대로 매매(상승=롱, 하락=숏, 중립=현금)했을 때 누적
    pos = d.sig.map({"상승": 1, "하락": -1, "중립": 0})
    eq = (1 + pos * d.y).prod() - 1
    bh = (1 + d.y).prod() - 1
    return by, acc, len(act), len(d), eq * 100, bh * 100


def split_check(rules, f, th):
    """앞 70%/뒤 30% 나눠 적중률 비교 → 과최적화 여부 확인"""
    cut = int(len(f) * 0.7)
    out = []
    for name, part in (("앞70%", f.iloc[:cut]), ("뒤30%", f.iloc[cut:])):
        _, s = score(rules, part)
        _, acc, n, *_ = score_report(s, part, th)
        out.append((name, acc, n))
    return out


# ---------------- 출력 ----------------
def run(px, cfg, html=None):
    rules, th = cfg["rules"], cfg["threshold"]
    f = features(px)
    hits, s = score(rules, f)
    rr, base = rule_report(rules, hits, f)
    by, acc, n_act, n_all, eq, bh = score_report(s, f, th)
    sp = split_check(rules, f, th)

    last = f.index[-1]
    today_hits = [r for r in rules if hits[r["name"]].iloc[-1]]
    sc = int(s.iloc[-1])
    sig = signal(pd.Series([sc]), th)[0]
    hist = by.loc[sc] if sc in by.index else None

    pd.set_option("display.width", 140)
    print(f"\n=== {last:%Y-%m-%d} 종가 기준 → 다음 거래일 예측 ===")
    print(f"점수 {sc:+d}  (기준 ±{th})  →  {sig}")
    if hist is not None:
        print(f"과거 같은 점수: {int(hist['일수'])}일, 상승 {hist['상승확률']:.0f}%, 평균 {hist['평균수익']:+.2f}%")
    for r in today_hits:
        print(f"  {r['points']:+d}  {r['name']}")
    print(f"\n=== 규칙별 검증 (전체 기간 상승일 비율 {base*100:.1f}%) ===")
    print(rr.to_string(index=False, float_format="%.1f"))
    print("\n=== 점수별 다음날 성적 ===")
    print(by.to_string(float_format="%.1f"))
    print(f"\n신호 적중률 {acc:.1f}% ({n_act}/{n_all}일 신호)")
    print(f"신호매매 누적 {eq:+.0f}%  vs  보유 {bh:+.0f}%  (수수료 제외)")
    print("기간 분할: " + " / ".join(f"{a} {b:.1f}% ({c}일)" for a, b, c in sp))

    if html:
        write_html(html, last, sc, th, sig, hist, today_hits, rr, by, acc, n_act, n_all, eq, bh, sp, base)
        print(f"\nHTML 저장: {html}")


def write_html(path, last, sc, th, sig, hist, today_hits, rr, by, acc, n_act, n_all, eq, bh, sp, base):
    color = {"상승": "#d32", "하락": "#16c", "중립": "#888"}[sig]
    hh = (f"과거 같은 점수 {int(hist['일수'])}일 · 상승 {hist['상승확률']:.0f}% · 평균 {hist['평균수익']:+.2f}%"
          if hist is not None else "과거 같은 점수 없음")
    hits_html = "".join(f"<div class='kv'><span>{r['name']}</span><b>{r['points']:+d}</b></div>"
                        for r in today_hits) or "<div class='sub'>해당 규칙 없음</div>"
    tbl = lambda df: df.to_html(index=False, float_format=lambda x: f"{x:.1f}", border=0)
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>나스닥 예측기</title><style>
body{{font-family:'Malgun Gothic',sans-serif;margin:0;background:#f4f6f9;color:#222}}
.wrap{{max-width:560px;margin:0 auto;padding:12px}} h2{{font-size:16px;margin:20px 4px 8px}}
.card{{background:#fff;border-radius:12px;padding:14px;margin-bottom:10px;box-shadow:0 1px 3px rgba(0,0,0,.07);font-size:14px;overflow-x:auto}}
.big{{font-size:30px;font-weight:800;color:{color}}} .sub{{color:#888;font-size:13px}}
.kv{{display:flex;justify-content:space-between;border-top:1px solid #f0f2f5;padding:5px 0}}
table{{border-collapse:collapse;width:100%;font-size:12px}} td,th{{padding:4px;border-bottom:1px solid #eee;text-align:right}}
td:first-child,th:first-child{{text-align:left}}
</style></head><body><div class="wrap">
<h2>📊 다음 거래일 나스닥 예측</h2>
<div class="card"><div class="sub">{last:%Y-%m-%d} 종가 기준 · 점수 {sc:+d} (기준 ±{th})</div>
<div class="big">{sig}</div><div class="sub">{hh}</div>{hits_html}</div>
<h2>✅ 검증</h2><div class="card">
<div class="kv"><span>신호 적중률</span><b>{acc:.1f}% ({n_act}/{n_all}일)</b></div>
<div class="kv"><span>신호매매 누적 / 보유</span><b>{eq:+.0f}% / {bh:+.0f}%</b></div>
{''.join(f"<div class='kv'><span>{a}</span><b>{b:.1f}% ({c}일)</b></div>" for a, b, c in sp)}
<div class="sub">전체 상승일 비율 {base*100:.1f}% · 수수료 제외</div></div>
<h2>📋 규칙별</h2><div class="card">{tbl(rr)}</div>
<h2>🔢 점수별</h2><div class="card">{tbl(by.reset_index().rename(columns={'s': '점수'}))}</div>
<div class="sub" style="text-align:center;margin:20px">참고용 · 투자 판단 책임은 본인에게 있습니다</div>
</div></body></html>"""
    Path(path).write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default=str(HERE / "rules.json"))
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--html")
    a = ap.parse_args()
    cfg = json.loads(Path(a.rules).read_text(encoding="utf-8"))
    px = demo_prices() if a.demo else load_prices(a.years)
    run(px, cfg, a.html)


if __name__ == "__main__":
    sys.exit(main())
