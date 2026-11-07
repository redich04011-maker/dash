"""출렁임 심화: (1) 우리 판은 출렁이는 밤에 더 버나? (2) 출렁임 예측 고도화 (3) 예측 출렁임으로 판 크기 조절 효과
입력: night_features.csv(나스닥 밤 499일, 20:00 KST 지표 + ②), in/NIGHTS_시간외_우리성과_밤별.csv(212밤)
네트워크 불필요 — 저장된 자료만 사용
"""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N = pd.read_csv("night_features.csv", parse_dates=["d"]).set_index("d").sort_index()
N["a2"] = N.r2.abs()
# HAR식 과거 출렁임 (그날 이전 밤들만): 직전 1·5·20밤 평균 |②|
N["h1"] = N.a2.shift(1); N["h5"] = N.a2.shift(1).rolling(5).mean(); N["h20"] = N.a2.shift(1).rolling(20).mean()
N["absgap"] = N.nq_gap.abs(); N["abs1920"] = N.nq_19_20.abs(); N["absqqq"] = N.QQQ_chg.abs()
N["absn225"] = N.n225.abs(); N["absdax"] = N.dax_gap.abs(); N["drop_prev"] = (N.p_nq <= -0.013).astype(int)
O = pd.read_csv("in/NIGHTS_시간외_우리성과_밤별.csv", dtype={"d": str})
O["d"] = pd.to_datetime(O.d); O = O.set_index("d")
J = O.join(N, how="inner")
print(f"우리 밤 {len(O)} 중 나스닥 밤과 연결 {len(J)}밤\n")

# (1) 우리 판 = a + b·② + c·|②|
X = np.c_[J.r2 * 100, J.a2 * 100]; y = J["우리_08시_판당"].values
m = LinearRegression().fit(X, y)
res = y - m.predict(X); se = np.sqrt(np.diag(np.linalg.inv(np.c_[np.ones(len(X)), X].T @ np.c_[np.ones(len(X)), X]) * res.var(ddof=3)))
print("[1] 우리 판당(%) = a + b·②(%) + c·|②|(%)")
print(f"    a {m.intercept_:+.3f} (±{se[0]:.3f})  b {m.coef_[0]:+.3f} (±{se[1]:.3f})  c {m.coef_[1]:+.3f} (±{se[2]:.3f})")
print("    → c>0 이고 ±의 2배보다 크면 '출렁이는 밤에 방향과 무관하게 더 번다'(볼록)")
J["vt"] = pd.qcut(J.a2, 3, labels=["작은 밤", "중간 밤", "큰 밤"])
print("\n    실제 |②| 3등분별 우리 판")
for k, g in J.groupby("vt", observed=True):
    up, dn = g[g.r2 > 0], g[g.r2 <= 0]
    print(f"    {k}: {len(g)}밤 판당 {g['우리_08시_판당'].mean():+.3f}% 이김 {(g['우리_08시_판당']>0).mean()*100:4.1f}% | "
          f"오른 밤 {len(up)} {up['우리_08시_판당'].mean():+.3f}% / 내린 밤 {len(dn)} {dn['우리_08시_판당'].mean():+.3f}%")

# (2) 출렁임 예측 walk-forward
sets = {
    "기존(지표만)": ["vix", "vix_term", "vix9_term", "ma20gap", "vol20", "p_nq", "n225", "dax_gap", "nq_gap", "mon", "ev_FOMC"],
    "+과거출렁임(HAR)": ["h1", "h5", "h20"],
    "+절대값": ["absgap", "abs1920", "absqqq", "absn225", "absdax", "drop_prev"],
}
cols = []; pred = {}
Nn = N.dropna(subset=["h20"]).copy()
print(f"\n[2] 출렁임 예측 (walk-forward, 나스닥 밤 {len(Nn)}일)")
for nm, add in sets.items():
    cols = cols + add
    X = Nn[cols].fillna(Nn[cols].median()).values; y = Nn.a2.values; P = np.full(len(y), np.nan)
    for s in range(150, len(y), 10):
        P[s:s+10] = make_pipeline(StandardScaler(), Ridge(alpha=20)).fit(X[:s], y[:s]).predict(X[s:s+10])
    k = ~np.isnan(P); q = pd.qcut(P[k], 3, labels=False)
    c = np.corrcoef(P[k], y[k])[0, 1]
    print(f"  {nm:16s} 상관 {c:+.2f} | 예측 하/중/상 3등분 실제 |②| " + " / ".join(f"{y[k][q==i].mean()*100:.2f}%" for i in range(3)))
    pred[nm] = P
Nn["vpred"] = pred["+절대값"]
k = Nn.vpred.notna()
half = Nn.index[k][k.sum() // 2]
for nm, mm in (("앞", k & (Nn.index < half)), ("뒤", k & (Nn.index >= half))):
    q = pd.qcut(Nn.vpred[mm], 3, labels=False)
    print(f"    {nm}절반: " + " / ".join(f"{Nn.a2[mm][q==i].mean()*100:.2f}%" for i in range(3)))

# (3) 우리 밤에 예측 출렁임 적용 (예측은 그날 이전 자료로만 만든 값)
J2 = O.join(Nn[["vpred", "a2", "r2"]], how="inner").dropna(subset=["vpred"])
J2["vq"] = pd.qcut(J2.vpred, 3, labels=["예측 작음", "예측 중간", "예측 큼"])
print(f"\n[3] 우리 밤 {len(J2)}밤 — 예측 출렁임 3등분별 (20:00에 아는 값)")
for k_, g in J2.groupby("vq", observed=True):
    print(f"    {k_}: {len(g)}밤 판당 {g['우리_08시_판당'].mean():+.3f}% 이김 {(g['우리_08시_판당']>0).mean()*100:4.1f}% 실제|②| {g.a2.mean()*100:.2f}% 판당 표준편차 {g['우리_08시_판당'].std():.2f}")
p = J2["우리_08시_판당"].values; vq = J2.vq.cat.codes.values
for nm, w in {"전부 1배": np.ones(len(p)), "예측 큼 1.5배": np.where(vq == 2, 1.5, 1), "예측 큼 0.5배": np.where(vq == 2, .5, 1),
              "예측 작음 0.5배": np.where(vq == 0, .5, 1), "예측 작음 0.5·큼 1.5": np.select([vq == 0, vq == 2], [.5, 1.5], 1)}.items():
    r = p * w; print(f"    {nm:20s} 합계 {r.sum():+7.1f}%  판당(투입 1당) {r.sum()/w.sum():+.3f}%  최악 {r.min():+.2f}%  손익 표준편차 {r.std():.2f}")
h2 = J2.index[len(J2) // 2]
for nm, mm in (("앞절반", J2.index < h2), ("뒤절반", J2.index >= h2)):
    g = J2[mm]; print(f"    {nm}: " + " / ".join(f"{k_} {gg['우리_08시_판당'].mean():+.2f}%({len(gg)})" for k_, gg in g.groupby('vq', observed=True)))
J2[["vpred"]].assign(vpred=lambda x: (x.vpred*100).round(3)).rename(columns={"vpred": "예측출렁임_%"}).join(
    J2[["vq"]].rename(columns={"vq": "등급"})).to_csv("our_nights_volpred.csv", encoding="utf-8-sig")
