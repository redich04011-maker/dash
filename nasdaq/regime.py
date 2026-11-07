"""추세(일봉)별 상/하 적중률: 상승세 / 하락세 / 횡보세
추세 판정은 전일 종가까지의 일봉만 사용:
  상승세: 종가 > MA20 AND MA20 5일 기울기 > 0
  하락세: 종가 < MA20 AND MA20 5일 기울기 < 0
  횡보세: 그 외
"""
import numpy as np, pandas as pd, yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from night import build, hourly, asof, ET

d = yf.download("NQ=F", period="4y", interval="1d", auto_adjust=True, progress=False)["Close"].squeeze()
d.index = pd.to_datetime(d.index).tz_localize(None)
ma20, ma60 = d.rolling(20).mean(), d.rolling(60).mean()
slope = ma20.pct_change(5)
reg = pd.Series("횡보세", index=d.index)
reg[(d > ma20) & (slope > 0)] = "상승세"
reg[(d < ma20) & (slope < 0)] = "하락세"
reg = reg.shift(1)  # 전일까지 정보로 오늘 추세 판정

df = build()
df["reg"] = reg.reindex(pd.to_datetime(df.index)).values
df = df.dropna(subset=["reg"])
y = df.yA > 0
g = df.nq_gap
print(f"{len(df)}거래일  추세분포: {df.reg.value_counts().to_dict()}\n")
print("[추세별 기본] 상승일 비율")
for r, s in df.groupby("reg"): print(f"  {r}: 상승 {(s.yA>0).mean()*100:.1f}% ({len(s)}일)")

print("\n[1] 밤8시 갭 방향 적중률 (추세별 × 갭크기)")
print("  갭≥     " + "".join(f"{r:>14}" for r in ("상승세", "횡보세", "하락세")))
for lo in (0, 0.0025, 0.005, 0.0075, 0.01):
    row = ""
    for r in ("상승세", "횡보세", "하락세"):
        m = (df.reg == r) & (g.abs() >= lo)
        row += f"{((g[m]>0)==y[m]).mean()*100:8.1f}%({m.sum():3d})" if m.sum() else " " * 14
    print(f"  {lo*100:.2f}%  {row}")

print("\n[1b] 추세와 같은 방향 갭 vs 반대 방향 갭 (|갭|≥0.25%)")
for r, sign in (("상승세", 1), ("하락세", -1)):
    m = (df.reg == r) & (g.abs() >= 0.0025)
    same = m & (np.sign(g) == sign); opp = m & (np.sign(g) == -sign)
    print(f"  {r}: 순방향 갭 {((g[same]>0)==y[same]).mean()*100:.1f}%({same.sum()}) | 역방향 갭 {((g[opp]>0)==y[opp]).mean()*100:.1f}%({opp.sum()})")

print("\n[2] 시각별 적중률 (추세별)")
nq = hourly("NQ=F")
for kst in ("20:00", "23:30", "00:30", "01:30"):
    res = {}
    for dt, row in df.iterrows():
        d0 = pd.Timestamp(dt) - pd.tseries.offsets.BDay(1)
        c0 = asof(nq, pd.Timestamp(f"{d0.date()} 16:00", tz=ET).tz_convert("UTC"))
        c1 = asof(nq, pd.Timestamp(f"{dt} 16:00", tz=ET).tz_convert("UTC"))
        b = pd.Timestamp(f"{dt} {kst}", tz="Asia/Seoul")
        if kst < "12:00": b += pd.Timedelta(days=1)
        p = asof(nq, b.tz_convert("UTC"))
        if np.isnan([c0, c1, p]).any(): continue
        res.setdefault(row.reg, []).append((p > c0) == (c1 > c0))
    print(f"  KST {kst}: " + " | ".join(f"{r} {np.mean(v)*100:.1f}%({len(v)})" for r, v in sorted(res.items())))

print("\n[3] 모델(+추세 변수) 확률구간별 적중 — walk-forward")
df["r_up"] = (df.reg == "상승세").astype(int); df["r_dn"] = (df.reg == "하락세").astype(int)
df["gap_x_up"] = df.nq_gap * df.r_up; df["gap_x_dn"] = df.nq_gap * df.r_dn
feats = [c for c in df.columns if c not in ("yA", "yB", "reg")]
X = df[feats].values.astype(float); yy = y.values.astype(int); P = np.full(len(yy), np.nan)
for s in range(200, len(yy), 20):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(X[:s], yy[:s])
    P[s:s+20] = m.predict_proba(X[s:s+20])[:, 1]
k = ~np.isnan(P); conf = np.abs(P - .5) + .5; hit = (P > .5) == yy.astype(bool)
for th in (0.5, 0.7, 0.8, 0.9):
    row = []
    for r in ("상승세", "횡보세", "하락세"):
        m = k & (conf >= th) & (df.reg.values == r)
        row.append(f"{r} {hit[m].mean()*100:.1f}%({m.sum()})" if m.sum() else f"{r} -")
    m = k & (conf >= th)
    print(f"  확률≥{th:.0%}: 전체 {hit[m].mean()*100:.1f}%({m.sum()}) | " + " | ".join(row))
