"""출렁임 보조지표 넓게 시험 (20:00 KST까지 정보만)
평가 1: 실제 출렁임 |②| 와의 순위상관, 상위20% 밤의 |②| (앞/뒤 절반)
평가 2: 우리 판 '그 밤 한 판 더' (문턱=앞 기간 상위20%에서 고정) → 앞/뒤/애프터 추가 이익 합
"""
import numpy as np, pandas as pd, yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pattern import nq, ET


def asof(s, ts):
    i = s.index.searchsorted(ts - pd.Timedelta(hours=1), side="right") - 1
    return s.iloc[i] if i >= 0 else np.nan


N = pd.read_csv("night_features.csv", parse_dates=["d"]).set_index("d").sort_index()
N["a2"] = N.r2.abs()
O = pd.read_csv("in/NIGHTS_시간외_우리성과_밤별.csv", dtype={"d": str}); O["d"] = pd.to_datetime(O.d); O = O.set_index("d")

# ---- 일봉 재료 ----
T = ["NQ=F", "^VIX", "^VXN", "^VVIX", "^MOVE", "^SKEW", "^KQ11", "^KS11", "KRW=X"]
raw = yf.download(T, period="4y", interval="1d", auto_adjust=True, progress=False)
C, H, L, Op = raw["Close"], raw["High"], raw["Low"], raw["Open"]
for x in (C, H, L, Op): x.index = pd.to_datetime(x.index).tz_localize(None)
c, h, l, o = C["NQ=F"], H["NQ=F"], L["NQ=F"], Op["NQ=F"]
tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
us = pd.DataFrame({                                        # 미국 일봉: 전 거래일 종가 기준 → 산 날 d에 shift(1)
    "atr14": tr.rolling(14).mean() / c,
    "bbw": c.rolling(20).std() * 4 / c.rolling(20).mean(),
    "park20": np.sqrt((np.log(h / l) ** 2).rolling(20).mean() / (4 * np.log(2))),
    "gk10": np.sqrt((0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2).rolling(10).mean()),
    "rv5": c.pct_change().rolling(5).std(), "rv20": c.pct_change().rolling(20).std(),
    "rv5_20": c.pct_change().rolling(5).std() / c.pct_change().rolling(20).std(),
    "range1": (h - l) / c,
    "vxn": C["^VXN"], "vvix": C["^VVIX"], "move": C["^MOVE"], "skew": C["^SKEW"],
    "vix_d1": C["^VIX"].pct_change(), "vix_d5": C["^VIX"].pct_change(5),
    "vix_pct": C["^VIX"].rolling(250, min_periods=120).rank(pct=True),
    "vxn_vix": C["^VXN"] / C["^VIX"],
}).shift(1)
kr = pd.DataFrame({                                        # 한국 일봉: 산 날 d 당일(15:30 마감, 20:00 전에 앎)
    "kq_ret": C["^KQ11"].pct_change(), "kq_rng": (H["^KQ11"] - L["^KQ11"]) / C["^KQ11"].shift(),
    "ks_rng": (H["^KS11"] - L["^KS11"]) / C["^KS11"].shift(),
    "kq_rv5": C["^KQ11"].pct_change().rolling(5).std(),
})
krw = C["KRW=X"].pct_change().rolling(5).std().shift(1).rename("krw_rv5")


# ---- 시간봉 재료 (NQ, 20:00 KST 직전까지) ----
def hfeat(d):
    T_ = pd.Timestamp(f"{d.date()} 11:00", tz="UTC")
    s = nq[(nq.index + pd.Timedelta(hours=1)) <= T_]
    r24 = s.iloc[-24:].pct_change().dropna(); r120 = s.iloc[-120:].pct_change().dropna()
    return {"hv24": r24.std(), "hv120": r120.std(), "hrng24": (s.iloc[-24:].max() - s.iloc[-24:].min()) / s.iloc[-1],
            "hv24_120": r24.std() / r120.std() if r120.std() else np.nan}


F = N.join(us, how="left").join(kr, how="left").join(krw, how="left")
F = F.join(pd.DataFrame([hfeat(d) for d in F.index], index=F.index))
F["h5"] = F.a2.shift(1).rolling(5).mean(); F["h20"] = F.a2.shift(1).rolling(20).mean()
F["absgap"] = F.nq_gap.abs(); F["neg_ma20"] = -F.ma20gap; F["neg_pnq"] = -F.p_nq
F["neg_n225"] = -F.n225; F["neg_kq"] = -F.kq_ret
F.to_csv("night_features2.csv")

NEW = ["atr14", "bbw", "park20", "gk10", "rv5", "rv20", "rv5_20", "range1", "vxn", "vvix", "move", "skew", "vix_d1", "vix_d5",
       "vix_pct", "vxn_vix", "kq_rng", "ks_rng", "kq_rv5", "neg_kq", "krw_rv5", "hv24", "hv120", "hrng24", "hv24_120"]
OLD = ["vix", "vix_term", "vix9_term", "neg_ma20", "vol20", "neg_pnq", "neg_n225", "absgap", "h5", "h20"]
half = F.index[len(F) // 2]; AFTER = pd.Timestamp("2026-09-15")
J = O.join(F, how="inner"); p = J["우리_08시_판당"]
JA, JB, JC = J.index < half, (J.index >= half) & (J.index < AFTER), J.index >= AFTER
print(f"나스닥 밤 {len(F)}일, 우리 밤 {len(J)}밤 (앞 {JA.sum()} / 뒤 {JB.sum()} / 애프터 {JC.sum()})")
print(f"지금(늘 1판) 합: 앞 {p[JA].sum():+.1f} / 뒤 {p[JB].sum():+.1f} / 애프터 {p[JC].sum():+.1f}\n")


def evaluate(f):
    x = F[f]; ok = x.notna() & F.a2.notna()
    A, B = ok & (F.index < half), ok & (F.index >= half)
    rho_a = x[A].rank().corr(F.a2[A].rank()); rho_b = x[B].rank().corr(F.a2[B].rank())
    th = x[A].quantile(.8)                                   # 문턱: 앞 기간 나스닥 밤 상위20%
    ta, tb = F.a2[A & (x >= th)].mean(), F.a2[B & (x >= th)].mean()
    xj = J[f]; m = xj >= th
    add = [p[m & JA].sum(), p[m & JB].sum(), p[m & JC].sum()]; n = [(m & JA).sum(), (m & JB).sum(), (m & JC).sum()]
    return rho_a, rho_b, ta, tb, add, n, th


rows = []
for f in OLD + NEW:
    if f not in F or F[f].notna().sum() < 200: continue
    ra, rb, ta, tb, add, n, th = evaluate(f)
    rows.append((f, ra, rb, ta, tb, add, n, f in NEW))
rows.sort(key=lambda r: -(r[1] + r[2]))
print(f"기준 |②|: 앞 {F.a2[F.index < half].mean()*100:.2f}% / 뒤 {F.a2[F.index >= half].mean()*100:.2f}%")
print(f"{'지표':10s} {'새':2s} {'상관앞':>6s} {'상관뒤':>6s} {'상위20%|②|앞':>11s} {'뒤':>6s} | 한판더 추가이익 앞(밤) / 뒤(밤) / 애프터(밤)  세 기간 모두 +")
for f, ra, rb, ta, tb, add, n, new in rows:
    allp = "★" if all(a > 0 for a in add) and min(n) >= 1 else ""
    print(f"{f:10s} {'새' if new else '  '} {ra:+.2f}  {rb:+.2f}  {ta*100:9.2f}%  {tb*100:5.2f}% | "
          f"{add[0]:+6.1f}({n[0]:2d}) / {add[1]:+6.1f}({n[1]:2d}) / {add[2]:+6.1f}({n[2]:2d})  {allp}")

# ---- 출렁임 예측 v2 (기존 + 새 보조지표) ----
def model(cols, tag):
    X = F[cols].copy(); X = X.fillna(X.expanding().median()).fillna(0).values; y = F.a2.values; P = np.full(len(y), np.nan)
    for s in range(150, len(y), 10):
        tr = ~np.isnan(y[:s])
        P[s:s+10] = make_pipeline(StandardScaler(), Ridge(alpha=30)).fit(X[:s][tr], y[:s][tr]).predict(X[s:s+10])
    k = ~np.isnan(P); P = pd.Series(P, index=F.index)
    th = P[(F.index < half) & k].quantile(.8)
    m = (J.index.isin(P.index)) & (P.reindex(J.index) >= th).values
    add = [p[m & JA].sum(), p[m & JB].sum(), p[m & JC].sum()]; n = [(m & JA).sum(), (m & JB).sum(), (m & JC).sum()]
    rho = P[k].rank().corr(F.a2[k].rank())
    print(f"  {tag:22s} 순위상관 {rho:+.2f} | 상위20% 한판더: 앞 {add[0]:+.1f}({n[0]}) / 뒤 {add[1]:+.1f}({n[1]}) / 애프터 {add[2]:+.1f}({n[2]})")
    return P


print("\n[출렁임 예측 모델 비교] (walk-forward, 문턱=앞 기간 예측 상위20%)")
P1 = model(["vix", "vix_term", "vix9_term", "ma20gap", "vol20", "p_nq", "n225", "dax_gap", "nq_gap", "mon", "ev_FOMC", "h5", "h20"], "기존(vol2와 같은 재료)")
good = [r[0] for r in rows if r[7] and (r[1] + r[2]) / 2 > .15]
P2 = model(["vix", "vix_term", "vix9_term", "ma20gap", "vol20", "p_nq", "n225", "dax_gap", "nq_gap", "mon", "ev_FOMC", "h5", "h20"] + good, "+새 보조지표(상관>0.15)")
print(f"    추가된 새 지표: {good}")
pd.DataFrame({"d": F.index.strftime("%Y%m%d"), "출렁임예측_기존_%": (P1 * 100).round(3), "출렁임예측_v2_%": (P2 * 100).round(3),
              "실제_|②|_%": (F.a2 * 100).round(3)}).to_csv("night_volpred_v2.csv", index=False, encoding="utf-8-sig")
print("저장: night_features2.csv, night_volpred_v2.csv")
