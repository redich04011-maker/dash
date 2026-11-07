"""밤 8시(KST) 나스닥 상/중/하 예측 — 실데이터 walk-forward 검증

기준시각 T = 매 거래일 11:00 UTC (= 한국 20:00)
T 이전 정보만 사용:
  - NQ 선물: 전일 16:00 ET → T 변화
  - DAX: 전일 종가 → T 변화
  - 닛케이/코스피 당일 등락 (T 전에 마감)
  - BTC: 전일 16:00 ET → T 변화
  - 전일 미국: NQ/SOX/VIX/TNX/HYG-IEF/달러
대상
  A) 전일 종가 → 당일 종가 (16:00 ET)
  B) T(밤 8시) → 당일 종가   ← 실제 매매 가능한 구간
등급: 상/중/하 = 최근 20일 변동성 σ 기준 ±0.43σ (대략 3등분)
"""
import sys
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ET = "America/New_York"
H = ["NQ=F", "^GDAXI", "BTC-USD"]
D = ["NQ=F", "^SOX", "^VIX", "^TNX", "HYG", "IEF", "DX-Y.NYB", "^N225", "^KS11", "^GDAXI"]


def hourly(t):
    df = yf.download(t, period="729d", interval="1h", auto_adjust=True, progress=False)
    s = df["Close"].squeeze().dropna()
    s.index = s.index.tz_convert("UTC")
    return s


def asof(s, ts):
    """ts 시점까지 '완성된' 1시간봉의 종가 (봉 시작 + 1h <= ts)"""
    sub = s[s.index + pd.Timedelta(hours=1) <= ts]
    return sub.iloc[-1] if len(sub) else np.nan


def build():
    hs = {t: hourly(t) for t in H}
    dl = yf.download(D, period="4y", interval="1d", auto_adjust=True, progress=False)["Close"]
    dl.index = pd.to_datetime(dl.index).tz_localize(None)
    nq = hs["NQ=F"]
    days = sorted({d for d in nq.index.tz_convert(ET).date})
    us = dl["NQ=F"].dropna().index  # 미국 거래일
    us = [d.date() for d in us if d.date() in set(days)]

    asia = {k: dl[k].dropna().pct_change() for k in ("^N225", "^KS11")}
    prev = pd.DataFrame({
        "p_nq": dl["NQ=F"].pct_change(), "p_sox": dl["^SOX"].pct_change(),
        "p_vix": dl["^VIX"].pct_change(), "p_tnx": dl["^TNX"].diff(),
        "p_hy": (dl["HYG"] / dl["IEF"]).pct_change(), "p_dxy": dl["DX-Y.NYB"].pct_change(),
        "p_nq5": dl["NQ=F"].pct_change(5),
        "p_nq_ma20": dl["NQ=F"] / dl["NQ=F"].rolling(20).mean() - 1,
        "vol20": dl["NQ=F"].pct_change().rolling(20).std(),
    }).ffill()
    dax_close = dl["^GDAXI"].dropna()

    rows = []
    for i in range(1, len(us)):
        d0, d1 = us[i - 1], us[i]
        c0 = pd.Timestamp(f"{d0} 16:00", tz=ET).tz_convert("UTC")
        c1 = pd.Timestamp(f"{d1} 16:00", tz=ET).tz_convert("UTC")
        T = pd.Timestamp(f"{d1} 11:00", tz="UTC")
        nq0, nqT, nq1 = asof(nq, c0), asof(nq, T), asof(nq, c1)
        b0, bT = asof(hs["BTC-USD"], c0), asof(hs["BTC-USD"], T)
        daxT = asof(hs["^GDAXI"], T)
        dpre = dax_close[dax_close.index.date < d1]
        dk = pd.Timestamp(d0)
        if dk not in prev.index or not len(dpre):
            continue
        r = dict(date=d1,
                 nq_gap=nqT / nq0 - 1,
                 btc_gap=bT / b0 - 1,
                 dax_gap=daxT / dpre.iloc[-1] - 1 if daxT == daxT else 0.0,
                 n225=asia["^N225"].get(pd.Timestamp(d1), 0.0),
                 kospi=asia["^KS11"].get(pd.Timestamp(d1), 0.0),
                 dow=pd.Timestamp(d1).dayofweek,
                 **prev.loc[dk].to_dict(),
                 yA=nq1 / nq0 - 1, yB=nq1 / nqT - 1)
        rows.append(r)
    df = pd.DataFrame(rows).set_index("date")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def grade(y, vol, k=0.43):
    return np.where(y > k * vol, 2, np.where(y < -k * vol, 0, 1))  # 0하 1중 2상


def walk_forward(X, y, make, start=200, step=20):
    pred = np.full(len(y), -1)
    prob = np.zeros(len(y))
    for s in range(start, len(y), step):
        m = make().fit(X[:s], y[:s])
        e = min(s + step, len(y))
        p = m.predict_proba(X[s:e])
        pred[s:e] = m.classes_[p.argmax(1)]
        prob[s:e] = p.max(1)
    return pred, prob


def report(name, y, pred, prob, labels):
    m = pred >= 0
    y, pred, prob = y[m], pred[m], prob[m]
    acc = (y == pred).mean() * 100
    base = pd.Series(y).value_counts(normalize=True).max() * 100
    out = [f"  {name}: 적중 {acc:.1f}% ({len(y)}일, 최빈값 찍기 {base:.1f}%)"]
    for q in (0.7, 0.85):
        th = np.quantile(prob, q)
        sel = prob >= th
        out.append(f"    확신 상위 {int((1-q)*100)}%: {(y[sel]==pred[sel]).mean()*100:.1f}% ({sel.sum()}일)")
    ct = pd.crosstab(pd.Series(y, name="실제").map(labels), pd.Series(pred, name="예측").map(labels))
    out.append("    " + ct.to_string().replace("\n", "\n    "))
    return "\n".join(out)


def main():
    df = build()
    feats = [c for c in df.columns if c not in ("yA", "yB")]
    X = df[feats].values
    print(f"데이터 {df.index[0]} ~ {df.index[-1]}  ({len(df)}거래일)\n")
    print("상관(지표 ↔ 대상A/B):")
    print(df[feats].corrwith(df.yA).to_frame("A").join(df[feats].corrwith(df.yB).to_frame("B"))
          .round(3).to_string(), "\n")

    models = {
        "로지스틱": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)),
        "부스팅": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                       max_iter=150, l2_regularization=1.0),
    }
    for tgt in ("yA", "yB"):
        title = "A) 전일종가→당일종가" if tgt == "yA" else "B) 밤8시→당일종가 (매매가능)"
        print(f"===== {title} =====")
        y3 = grade(df[tgt].values, df.vol20.values)
        y2 = (df[tgt].values > 0).astype(int)
        if tgt == "yA":  # 단순 기준: NQ 갭 방향
            g = (df.nq_gap.values > 0).astype(int)
            print(f"  [기준] NQ 갭 방향만: 상/하 {(g==y2)[200:].mean()*100:.1f}%")
        for mn, mk in models.items():
            p3, pr3 = walk_forward(X, y3, mk)
            p2, pr2 = walk_forward(X, y2, mk)
            print(f" [{mn}]")
            print(report("상중하", y3, p3, pr3, {0: "하", 1: "중", 2: "상"}))
            print(report("상하", y2, p2, pr2, {0: "하", 1: "상"}))
        print()

    # 오늘 밤 예측 (마지막 행 = 가장 최근 거래일 기준 학습 후)
    last = df.iloc[[-1]][feats].values
    for tgt in ("yA", "yB"):
        y3 = grade(df[tgt].values, df.vol20.values)
        m = models["로지스틱"]().fit(X, y3)
        p = m.predict_proba(last)[0]
        print(f"최근일 {df.index[-1]} {tgt} 확률 하/중/상 = " + "/".join(f"{v*100:.0f}%" for v in p))


if __name__ == "__main__":
    sys.exit(main())
