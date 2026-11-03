"""목표②: 20:00 KST 매수 → 다음날 08:00 KST 의 NQ선물 방향 (8시 정보만 사용)
C = NQ(다음날 08:00 KST = 23:00 UTC 당일 이전 완성봉) / NQ(20:00 KST = 11:00 UTC) - 1
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pattern import df, gap, nq, asof, P, shape
from night import build

d = df.copy(); d.index = pd.DatetimeIndex(d.date); d["gap"] = gap
C = []
for dt in d.index:
    T = pd.Timestamp(f"{dt.date()} 11:00", tz="UTC"); E = pd.Timestamp(f"{dt.date()} 23:00", tz="UTC")
    if E > pd.Timestamp.now(tz="UTC"): C.append(np.nan); continue
    C.append(asof(nq, E) / asof(nq, T) - 1)
d["C"] = C
b = build(); b.index = pd.to_datetime(b.index)
d = d.join(b[["dax_gap", "n225", "kospi", "btc_gap", "p_nq", "p_nq5"]], how="left")
d["mom3"] = [p[-1] - p[-4] for p in d.pre]                 # 20시 직전 3시간 흐름
d["rng"] = [p.max() - p.min() for p in d.pre]
d["fromhi"] = [p[-1] - p.max() for p in d.pre]
d["fromlo"] = [p[-1] - p.min() for p in d.pre]
d["mon"] = (d.index.dayofweek == 0).astype(int); d["fri"] = (d.index.dayofweek == 4).astype(int)
ok = d.C.notna() & d[["dax_gap", "n225", "kospi", "btc_gap", "p_nq"]].notna().all(1)
D = d[ok].copy(); up = (D.C > 0).values
print(f"{D.index[0].date()} ~ {D.index[-1].date()} {len(D)}일 | ② 상승 비율 {up.mean()*100:.1f}% | 평균 {D.C.mean()*100:+.3f}% | 평균|②| {D.C.abs().mean()*100:.2f}%\n")

feats = ["gap", "mom3", "rng", "fromhi", "fromlo", "reg", "ma20gap", "vix", "vol20", "dax_gap", "n225", "kospi",
         "btc_gap", "p_nq", "p_nq5", "mon", "fri", "mstart"]
print("[상관: 지표 ↔ ②]")
print("  " + "  ".join(f"{f} {np.corrcoef(D[f].astype(float), D.C)[0,1]:+.3f}" for f in feats))

def acc(sig, m=None):
    m = np.ones(len(D), bool) if m is None else m
    s = np.sign(sig[m]); k = s != 0
    return ((s[k] > 0) == up[m][k]).mean() * 100, k.sum(), (s[k] * D.C.values[m][k]).mean() * 100

print("\n[단순 규칙] 적중 / 일수 / 건당 평균손익")
rules = {
    "항상 매수": np.ones(len(D)),
    "갭 따라가기": D.gap.values, "갭 반대(되돌림)": -D.gap.values,
    "직전3시간 따라가기": D.mom3.values, "직전3시간 반대": -D.mom3.values,
    "추세 따라(상승세 매수/하락세 매도)": D.reg.values.astype(float),
    "20일선 이격 반대": -D.ma20gap.values,
}
for nm, s in rules.items():
    a, n, r = acc(s); print(f"  {nm:30s} {a:5.1f}% {n:4d}일 {r:+.3f}%")
for lo in (.005, .0075, .01):
    m = D.gap.abs().values >= lo
    a, n, r = acc(-D.gap.values, m); print(f"  갭 반대, |갭|≥{lo*100:.2f}%            {a:5.1f}% {n:4d}일 {r:+.3f}%")
for nm, m in {"상승세": D.reg.values == 1, "횡보세": D.reg.values == 0, "하락세": D.reg.values == -1,
              "월요일": D.mon.values == 1, "금요일": D.fri.values == 1}.items():
    a, n, r = acc(np.ones(len(D)), m); print(f"  {nm}에 항상 매수              {a:5.1f}% {n:4d}일 {r:+.3f}%")

# walk-forward 로지스틱
X = D[feats].values.astype(float); Pm = np.full(len(D), np.nan)
for s in range(200, len(D), 20):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X[:s], up[:s])
    Pm[s:s + 20] = m.predict_proba(X[s:s + 20])[:, 1]
k = ~np.isnan(Pm); hit = (Pm > .5) == up; cf = np.abs(Pm - .5) + .5; ret = np.sign(Pm - .5) * D.C.values
print(f"\n[모델 walk-forward {k.sum()}일] 적중 {hit[k].mean()*100:.1f}% / 건당 {ret[k].mean()*100:+.3f}%  (같은 기간 항상매수 {up[k].mean()*100:.1f}%)")
for th in (.55, .6, .65, .7):
    m = k & (cf >= th)
    if m.sum(): print(f"  확신≥{int(th*100)}%: {hit[m].mean()*100:5.1f}% ({m.sum()}일) 건당 {ret[m].mean()*100:+.3f}%")

# 패턴 KNN (8시까지 모양 → ② 방향)
Ps = P[d.index.get_indexer(D.index)]; Sh = shape[d.index.get_indexer(D.index)]
def knn(X_, k_=30):
    pr = np.full(len(D), np.nan)
    for i in range(150, len(D)):
        dist = np.sqrt(((X_[:i] - X_[i]) ** 2).sum(1)); nb = np.argsort(dist)[:k_]
        pr[i] = up[nb].mean()
    return pr
for nm, X_ in (("모양+크기", Ps), ("모양만", Sh)):
    pr = knn(X_); kk = ~np.isnan(pr); h = (pr > .5) == up; c_ = np.abs(pr - .5) + .5
    print(f"\n[패턴 {nm} K=30, {kk.sum()}일] 적중 {h[kk].mean()*100:.1f}% (같은 기간 항상매수 {up[kk].mean()*100:.1f}%)")
    for th in (.6, .7):
        m = kk & (c_ >= th)
        if m.sum(): print(f"  일치율≥{int(th*100)}%: {h[m].mean()*100:5.1f}% ({m.sum()}일) 건당 {(np.sign(pr-.5)*D.C.values)[m].mean()*100:+.3f}%")

print("\n[최근 15일] 날짜 갭 / 직전3h / 모델 상승확률 → ② 실제")
for i in range(len(D) - 15, len(D)):
    print(f"  {D.index[i].date()} 갭 {D.gap.iloc[i]*100:+.2f}% 3h {D.mom3.iloc[i]*100:+.2f}% 모델 {Pm[i]*100:4.0f}% → ② {D.C.iloc[i]*100:+.2f}% {'O' if hit[i] else 'X'}")
