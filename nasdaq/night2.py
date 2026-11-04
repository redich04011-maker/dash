"""시간외창 의뢰(2026-11-04) — 밤 구간 연구
정보 시점: 산 날 d 의 20:00 KST(=11:00 UTC) 까지만.
밤 ② = NQ(d2 08:00 KST) / NQ(d 20:00 KST) - 1
목표  1-A 우리 판 이김(우리_08시_판당>0)  1-B 큰 하락(②≤-0.7%, ≤-1%)  1-C |②|
검증  앞절반에서 고르고 → 뒤절반·애프터(2026-09-15~) 따로 / 이웃 칸 고원 / 뒤섞기 시험 / 30일 미만 '판정 불가'
"""
import time, json
import numpy as np, pandas as pd, requests, yfinance as yf
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pattern import df, gap, nq, ET
import sys


def asof(s, ts):
    """ts 시점까지 완성된 1시간봉 종가 (빠른 버전)"""
    i = s.index.searchsorted(ts - pd.Timedelta(hours=1), side="right") - 1
    return s.iloc[i] if i >= 0 else np.nan


def log(*a):
    print(*a, flush=True)
from night import build

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
      "Accept": "application/json, text/plain, */*"}
rng = np.random.default_rng(0)

# ---------------- 미국 거래일 표, 기본 지표 ----------------
base = df.copy(); base.index = pd.DatetimeIndex(base.date); base["gap"] = gap
b = build(); b.index = pd.to_datetime(b.index)
base = base.join(b[["dax_gap", "n225", "kospi", "btc_gap", "p_nq"]], how="left")
usd = list(base.index)


def prev_us(d):
    p = [x for x in usd if x < d]
    return p[-1] if p else None


# ---------------- 시간봉 재료 (프리마켓 포함) ----------------
def h1(t, pre=False):
    try:
        x = yf.download(t, period="729d", interval="1h", prepost=pre, auto_adjust=True, progress=False)["Close"].squeeze().dropna()
        x.index = x.index.tz_convert("UTC"); return x
    except Exception as e:
        print("  !", t, e); return pd.Series(dtype=float)


H = {"QQQ": h1("QQQ", True), "SMH": h1("SMH", True), "SOXL": h1("SOXL", True),
     "ZN": h1("ZN=F"), "DXY": h1("DX-Y.NYB"), "KRW": h1("KRW=X"), "GOLD": h1("GC=F"), "OIL": h1("CL=F"), "BTC": h1("BTC-USD")}
dv = yf.download(["^VIX", "^VIX3M", "^VIX9D"], period="4y", interval="1d", auto_adjust=True, progress=False)["Close"]
dv.index = pd.to_datetime(dv.index).tz_localize(None)


def feat_at(d):
    """d(산 날, KST 날짜) 20:00 KST 기준 지표"""
    d = pd.Timestamp(d); p = prev_us(d)
    if p is None: return None
    c0 = pd.Timestamp(f"{p.date()} 16:00", tz=ET).tz_convert("UTC"); T = pd.Timestamp(f"{d.date()} 11:00", tz="UTC")
    T19 = T - pd.Timedelta(hours=1)
    r = {"d": d}
    for k, s in H.items():
        if len(s):
            a, z = asof(s, c0), asof(s, T)
            r[f"{k}_chg"] = (z - a) if k == "ZN" else (z / a - 1)
    for k in ("QQQ", "SMH"):
        s = H[k]
        if len(s): r[f"{k}_19_20"] = asof(s, T) / asof(s, T19) - 1        # 19~20시 모양
    n0, nT, n19 = asof(nq, c0), asof(nq, T), asof(nq, T19)
    r["nq_gap"] = nT / n0 - 1; r["nq_19_20"] = nT / n19 - 1
    v = dv[dv.index < d]
    if len(v):
        r["vix"] = v["^VIX"].iloc[-1]; r["vix_term"] = v["^VIX"].iloc[-1] / v["^VIX3M"].iloc[-1]
        r["vix9_term"] = v["^VIX9D"].iloc[-1] / v["^VIX"].iloc[-1]
    if p in base.index:
        for c in ("reg", "ma20gap", "vol20", "mstart", "dax_gap", "n225", "kospi", "p_nq"):
            r[c] = base.at[d, c] if d in base.index and c in ("dax_gap", "n225", "kospi", "mstart") else base.at[p, c] if c in ("p_nq",) else (base.at[d, c] if d in base.index else base.at[p, c])
    r["mon"] = int(d.dayofweek == 0); r["fri"] = int(d.dayofweek == 4)
    return r


def night_ret(d, d2):
    T = pd.Timestamp(f"{pd.Timestamp(d).date()} 11:00", tz="UTC")
    E = pd.Timestamp(f"{pd.Timestamp(d2).date()} 08:00", tz="Asia/Seoul").tz_convert("UTC")
    return asof(nq, E) / asof(nq, T) - 1


# ---------------- 밤 일정 (나스닥 경제달력) ----------------
def econ(dates):
    out = {}
    keys = {"CPI": ["CPI", "Consumer Price Index"], "PPI": ["PPI", "Producer Price"], "NFP": ["Nonfarm Payrolls", "Non-Farm"],
            "FOMC": ["Fed Interest Rate Decision", "FOMC Statement", "Interest Rate Decision"], "PCE": ["PCE"], "GDP": ["GDP"],
            "소매": ["Retail Sales"]}
    ok = fail = 0
    for n, d in enumerate(dates):
        if n == 12 and ok == 0:
            log("  ! 경제달력 접속 불가 → 일정 없이 진행"); return {x: None for x in dates}
        if n % 100 == 0: log(f"  경제달력 {n}/{len(dates)}")
        try:
            r = requests.get("https://api.nasdaq.com/api/calendar/economicevents", params={"date": d.strftime("%Y-%m-%d")},
                             headers=UA, timeout=8)
            rows = (r.json().get("data") or {}).get("rows") or []
            ok += 1
            tags = set()
            for x in rows:
                if "United States" not in (x.get("country") or ""): continue
                nm = x.get("eventName") or ""
                for k, ws in keys.items():
                    if any(w.lower() in nm.lower() for w in ws): tags.add(k)
            out[d] = tags
        except Exception:
            out[d] = None
        time.sleep(0.25)
    print(f"  경제달력: {ok}/{len(dates)}일 수신")
    return out


# ================= 1) 나스닥 밤 표본 (전체 미국 거래일) =================
log("나스닥 밤 표본 계산"); rows = []
for d in usd[-500:]:
    f = feat_at(d)
    if not f: continue
    nx = [x for x in usd if x > d]
    if not nx: continue
    f["r2"] = night_ret(d, nx[0]); rows.append(f)
N = pd.DataFrame(rows).set_index("d")
N = N[N.r2.notna()]
log("경제달력 수집"); cal = econ(list(N.index))
for k in ("CPI", "PPI", "NFP", "FOMC", "PCE", "GDP", "소매"):
    N[f"ev_{k}"] = [int(k in (cal.get(d) or set())) for d in N.index]
N["ev_macro"] = N[["ev_CPI", "ev_PPI", "ev_NFP", "ev_FOMC", "ev_PCE"]].max(1)
cal_ok = sum(v is not None for v in cal.values())
N.to_csv("night_features.csv")

# ================= 2) 우리 성과 212밤 =================
O = pd.read_csv("in/NIGHTS_시간외_우리성과_밤별.csv", dtype={"d": str, "d2": str})
O["d"] = pd.to_datetime(O.d); O["d2"] = pd.to_datetime(O.d2)
orow = []
for _, x in O.iterrows():
    f = feat_at(x.d)
    if not f: continue
    f.update(r2=night_ret(x.d, x.d2), pnl=x["우리_08시_판당"], winr=x["이김비율"], 판수=x["판수"], 기간=x["기간"])
    ev = cal.get(x.d)
    for k in ("CPI", "PPI", "NFP", "FOMC", "PCE"):
        f[f"ev_{k}"] = int(k in (ev or set()))
    f["ev_macro"] = max(f[f"ev_{k}"] for k in ("CPI", "PPI", "NFP", "FOMC", "PCE"))
    orow.append(f)
U = pd.DataFrame(orow).set_index("d")
U["win"] = (U.pnl > 0).astype(int)
print(f"\n[자료] 나스닥 밤 {len(N)}일({N.index[0].date()}~{N.index[-1].date()}), 우리 밤 {len(U)}밤, 경제달력 수신 {cal_ok}일")
print(f"  우리 판 이김 비율 {U.win.mean()*100:.1f}%  판당 평균 {U.pnl.mean():+.3f}%  | 상관(우리 판당, ②) {np.corrcoef(U.pnl, U.r2)[0,1]:+.2f}")
for nm, m in (("② 상승 밤", U.r2 > 0), ("② 하락 밤", U.r2 <= 0), ("② ≤ -0.7%", U.r2 <= -.007), ("② ≤ -1%", U.r2 <= -.01)):
    print(f"  {nm:10s} {m.sum():3d}밤: 우리 이김 {U.win[m].mean()*100:5.1f}%  판당 {U.pnl[m].mean():+.3f}%")

# ================= 3) 규칙 스윕 =================
FEATS = ["nq_gap", "nq_19_20", "QQQ_chg", "QQQ_19_20", "SMH_chg", "SMH_19_20", "SOXL_chg", "ZN_chg", "DXY_chg", "KRW_chg",
         "GOLD_chg", "OIL_chg", "BTC_chg", "vix", "vix_term", "vix9_term", "ma20gap", "vol20", "dax_gap", "n225", "kospi", "p_nq"]
EV = ["ev_CPI", "ev_PPI", "ev_NFP", "ev_FOMC", "ev_PCE", "ev_macro", "mon", "fri", "mstart"]
AFTER = pd.Timestamp("2026-09-15")


def sweep(X, y, metric, title, higher_better=True, min_n=30):
    """앞절반에서 고르고 뒤절반·애프터 확인. metric(y_sub) -> 값"""
    X = X.copy(); n = len(X); half = X.index[n // 2]
    A, B = X.index < half, (X.index >= half) & (X.index < AFTER)
    C = X.index >= AFTER
    base_all = metric(y)
    res = []
    for f in FEATS:
        if f not in X or X[f].isna().mean() > .3: continue
        v = X[f].values
        qs = np.nanquantile(v[A], [.1, .2, .3, .7, .8, .9])
        for qi, q in enumerate(qs):
            side = "≤" if qi < 3 else "≥"
            m = (v <= q) if side == "≤" else (v >= q)
            m = m & ~np.isnan(v)
            res.append((f"{f} {side} {q:+.4g}", f, qi, m))
    for f in EV:
        if f in X: res.append((f"{f}=1", f, -1, X[f].values == 1))
    out = []
    for nm, f, qi, m in res:
        if m[A].sum() < 15: continue
        sa = metric(y[m & A]); out.append((nm, f, qi, m, sa))
    out.sort(key=lambda t: -(t[4] if higher_better else -t[4]))
    best_a = out[0][4] if out else np.nan
    # 뒤섞기: 앞절반에서 최고값이 우연으로 얼마나 나오나
    sh = []
    yA = y[A]
    for _ in range(200):
        yp = rng.permutation(yA); bests = []
        for nm, f, qi, m, sa in out[:400]:
            mm = m[A]
            if mm.sum() >= 15: bests.append(metric(yp[mm]))
        sh.append(max(bests) if higher_better else min(bests))
    p = np.mean([s >= best_a if higher_better else s <= best_a for s in sh])
    print(f"\n[{title}] 전체 기준 {base_all*100:.1f} / 앞절반 기준 {metric(y[A])*100:.1f} / 뒤 {metric(y[B])*100:.1f} / 애프터 {metric(y[C])*100:.1f}")
    print(f"  뒤섞기 시험: 앞절반 최고값 {best_a*100:.1f} 이상이 우연히 나올 확률 p={p:.2f} (0.05 이하면 우연 아님)")
    print(f"  {'규칙':28s} {'앞(일)':>11s} {'뒤(일)':>11s} {'애프터(일)':>11s} 고원")
    shown = 0
    for nm, f, qi, m, sa in out[:12]:
        sb, sc = (metric(y[m & B]) if (m & B).sum() else np.nan), (metric(y[m & C]) if (m & C).sum() else np.nan)
        nb, nc = (m & B).sum(), (m & C).sum()
        neigh = [t for t in out if t[1] == f and qi >= 0 and abs(t[2] - qi) == 1 and (t[2] < 3) == (qi < 3)]
        plat = "예" if neigh and all((t[4] - metric(y[A])) * (1 if higher_better else -1) > 0 for t in neigh) else ("-" if qi < 0 else "아니오")
        tag = lambda s, n_: f"{s*100:5.1f}({n_:3d})" + ("*" if n_ < min_n else " ")
        print(f"  {nm:28s} {tag(sa, (m & A).sum())} {tag(sb, nb) if nb else '    -      '} {tag(sc, nc) if nc else '    -      '} {plat}")
        shown += 1
    print("  (* = 30일 미만, 판정 불가)")
    return out


# 1-A 우리 판 이김
sweep(U[FEATS + [e for e in EV if e in U]], U.win.values.astype(float), np.mean, "1-A 우리 판 이김 비율(%)")
# 1-B 큰 하락 (나스닥 전체 표본, 정밀도 = 걸린 밤 중 진짜 큰 하락 비율)
for th in (.007, .01):
    sweep(N[FEATS + EV], (N.r2.values <= -th).astype(float), np.mean, f"1-B ② ≤ -{th*100:.1f}% 밤 비율(정밀도, %)")
# 1-C 출렁임 크기
sweep(N[FEATS + EV], np.abs(N.r2.values), np.mean, "1-C |②| 평균(%)")
# ② 방향 (참고)
sweep(N[FEATS + EV], (N.r2.values > 0).astype(float), np.mean, "② 상승 비율(%)")

# ================= 4) 모델(walk-forward) + 날짜별 예측 csv =================
F2 = [f for f in FEATS + EV if f in N and N[f].isna().mean() < .3]
Xn = N[F2].fillna(N[F2].median()).values
yb = (N.r2.values <= -.007).astype(int); ya = np.abs(N.r2.values); yu = (N.r2.values > 0).astype(int)
Pb, Pa, Pu = (np.full(len(N), np.nan) for _ in range(3))
for s in range(200, len(N), 20):
    Pb[s:s+20] = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=3000)).fit(Xn[:s], yb[:s]).predict_proba(Xn[s:s+20])[:, 1]
    Pu[s:s+20] = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=3000)).fit(Xn[:s], yu[:s]).predict_proba(Xn[s:s+20])[:, 1]
    Pa[s:s+20] = make_pipeline(StandardScaler(), Ridge(alpha=30)).fit(Xn[:s], ya[:s]).predict(Xn[s:s+20])
k = ~np.isnan(Pb)
print(f"\n[모델 walk-forward {k.sum()}일]")
br = yb[k].mean()
for q in (.8, .9):
    th = np.nanquantile(Pb[k], q); m = k & (Pb >= th)
    print(f"  큰하락확률 상위 {int((1-q)*100)}%: 실제 큰하락 {yb[m].mean()*100:.1f}% ({m.sum()}일) vs 기준 {br*100:.1f}%")
c = np.corrcoef(Pa[k], ya[k])[0, 1]
print(f"  |②| 예측-실제 상관 {c:+.2f}  | 예측 상위20% 실제 |②| {ya[k & (Pa >= np.nanquantile(Pa[k], .8))].mean()*100:.2f}% vs 하위20% {ya[k & (Pa <= np.nanquantile(Pa[k], .2))].mean()*100:.2f}%")
cf = np.abs(Pu - .5) + .5
for th in (.6, .65):
    m = k & (cf >= th)
    print(f"  방향 확신≥{int(th*100)}%: {((Pu[m] > .5) == yu[m]).mean()*100:.1f}% ({m.sum()}일)")
pred = pd.DataFrame({"d": N.index.strftime("%Y%m%d"), "상승확률": Pu.round(3), "큰하락확률_0.7": Pb.round(3),
                     "출렁임예측_abs": Pa.round(5), "방향": np.where(Pu > .5, "상", "하"), "확신": cf.round(3),
                     "실제_②": N.r2.round(5).values})
pred.to_csv("night_preds.csv", index=False, encoding="utf-8-sig")
# 우리 밤에 붙이기
U2 = U.join(pred.assign(d=pd.to_datetime(pred.d)).set_index("d")[["상승확률", "큰하락확률_0.7", "출렁임예측_abs"]], how="left")
kk = U2["큰하락확률_0.7"].notna()
if kk.sum() > 20:
    top = kk & (U2["큰하락확률_0.7"] >= U2["큰하락확률_0.7"][kk].quantile(.8))
    print(f"\n[우리 밤 {kk.sum()}밤에 모델 적용] 큰하락확률 상위20% {top.sum()}밤: 우리 이김 {U2.win[top].mean()*100:.1f}% 판당 {U2.pnl[top].mean():+.3f}% "
          f"| 나머지 이김 {U2.win[kk & ~top].mean()*100:.1f}% 판당 {U2.pnl[kk & ~top].mean():+.3f}%")
    print(f"  → '상위20% 밤만 반 매수' 가정 시 판당 평균 {np.where(top, U2.pnl*.5, U2.pnl)[kk].mean():+.3f}% vs 전부 매수 {U2.pnl[kk].mean():+.3f}%")

# ================= 5) MNQ 헤지 (조사용 수치) =================
beta = np.cov(U.pnl, U.r2 * 100)[0, 1] / np.var(U.r2 * 100)
print(f"\n[MNQ 헤지 참고] 우리 판당(%) = {beta:.2f} × ②(%) + α  (상관 {np.corrcoef(U.pnl, U.r2)[0,1]:+.2f}, 결정계수 {np.corrcoef(U.pnl, U.r2)[0,1]**2:.2f})")
print(f"  NQ 현재 {nq.iloc[-1]:,.0f} → MNQ 1계약 명목 ≈ ${nq.iloc[-1]*2:,.0f} (지수×$2)")
