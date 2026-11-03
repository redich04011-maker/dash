"""상/하 90% 가능 조건 탐색: (1) 갭 크기별 (2) 예측 시각별 (3) 갭+모델 확신"""
import numpy as np, pandas as pd, yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from night import build, hourly, asof, ET

df = build()
y = (df.yA > 0).astype(int).values
g = df.nq_gap.values
print(f"{len(df)}거래일\n\n[1] 밤8시 NQ갭 크기별 '갭 방향 = 종가 방향' 적중률")
for lo in (0, 0.0025, 0.005, 0.0075, 0.01, 0.015):
    m = np.abs(g) >= lo
    print(f"  |갭|≥{lo*100:.2f}%: {((g[m]>0)==y[m].astype(bool)).mean()*100:5.1f}%  ({m.sum()}일, {m.mean()*100:.0f}%의 날)")

print("\n[2] 예측 시각별 (전일종가 대비 그 시각 NQ 방향 = 종가 방향)")
nq = hourly("NQ=F")
for kst in ("20:00", "22:30", "23:30", "00:30", "01:30", "02:30", "03:30", "04:30"):
    hits = sig = 0; big = bh = 0
    for d in df.index:
        d0 = pd.Timestamp(d) - pd.tseries.offsets.BDay(1)
        c0 = asof(nq, pd.Timestamp(f"{d0.date()} 16:00", tz=ET).tz_convert("UTC"))
        c1 = asof(nq, pd.Timestamp(f"{d} 16:00", tz=ET).tz_convert("UTC"))
        base = pd.Timestamp(f"{d} {kst}", tz="Asia/Seoul")
        if kst < "12:00": base += pd.Timedelta(days=1)
        p = asof(nq, base.tz_convert("UTC"))
        if np.isnan([c0, c1, p]).any(): continue
        r = p / c0 - 1; ok = (r > 0) == (c1 > c0)
        sig += 1; hits += ok
        if abs(r) >= 0.005: big += 1; bh += ok
    print(f"  KST {kst}: 전체 {hits/sig*100:5.1f}% ({sig}일) | |변화|≥0.5%: {bh/max(big,1)*100:5.1f}% ({big}일)")

print("\n[3] 밤8시 모델 확률 구간별 (walk-forward 로지스틱)")
feats = [c for c in df.columns if c not in ("yA", "yB")]
X = df[feats].values; P = np.full(len(y), np.nan)
for s in range(200, len(y), 20):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(X[:s], y[:s])
    P[s:s+20] = m.predict_proba(X[s:s+20])[:, 1]
k = ~np.isnan(P); P, yy = P[k], y[k]
conf = np.abs(P - 0.5) + 0.5
for th in (0.6, 0.7, 0.8, 0.85, 0.9):
    m = conf >= th
    if m.sum(): print(f"  확률≥{th:.0%}: {(((P[m]>0.5)==yy[m].astype(bool)).mean()*100):5.1f}%  ({m.sum()}/{len(yy)}일)")
