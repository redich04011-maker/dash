"""시간외 표와 같은 방식으로 재채점: 우리 밤을 앞/뒤(애프터 전 절반씩)/애프터로 나눔.
'그 밤 한 판 더' = 그 밤 판당만큼 추가 이익. 문턱 = 앞 기간 우리 밤에서 지표 상위 20%(고정)."""
import numpy as np, pandas as pd
F = pd.read_csv("night_features2.csv", parse_dates=["d"]).set_index("d")
V = pd.read_csv("night_volpred_v2.csv", dtype={"d": str}); V["d"] = pd.to_datetime(V.d); V = V.set_index("d")
F = F.join(V[["출렁임예측_기존_%", "출렁임예측_v2_%"]])
O = pd.read_csv("in/NIGHTS_시간외_우리성과_밤별.csv", dtype={"d": str}); O["d"] = pd.to_datetime(O.d); O = O.set_index("d")
J = O.join(F, how="inner"); p = J["우리_08시_판당"]
AFTER = pd.Timestamp("2026-09-15"); pre = J.index[J.index < AFTER]; mid = pre[len(pre) // 2]
A, B, C = J.index < mid, (J.index >= mid) & (J.index < AFTER), J.index >= AFTER
print(f"우리 밤 {len(J)} (앞 {A.sum()} ~{mid.date()} / 뒤 {B.sum()} / 애프터 {C.sum()})")
print(f"지금(늘 1판) 합: 앞 {p[A].sum():+.1f} / 뒤 {p[B].sum():+.1f} / 애프터 {p[C].sum():+.1f}  (최악 {p.min():+.2f})\n")
J["nk_dax_vix"] = ((J.n225 < 0) & (J.dax_gap < 0) & (J.vix >= 16)).astype(float)
feats = ["출렁임예측_v2_%", "출렁임예측_기존_%", "vix", "vxn", "vvix", "move", "vix_term", "vix9_term", "vix_pct", "vix_d1", "vix_d5",
         "atr14", "park20", "gk10", "rv5", "rv20", "range1", "hv24", "hv120", "hrng24", "h5", "h20",
         "neg_ma20", "neg_pnq", "neg_n225", "neg_kq", "kq_rng", "ks_rng", "kq_rv5", "krw_rv5", "absgap", "vol20"]
rows = []
for f in feats:
    x = J[f]
    if x.notna().sum() < 150: continue
    th = x[A].quantile(.8); m = (x >= th).values
    add = [p[m & A].sum(), p[m & B].sum(), p[m & C].sum()]; n = [(m & A).sum(), (m & B).sum(), (m & C).sum()]
    rows.append((f, th, add, n, sum(add[1:])))
m = J.nk_dax_vix.values == 1
rows.append(("닛케이·DAX하락+VIX≥16", 1, [p[m & A].sum(), p[m & B].sum(), p[m & C].sum()], [(m & A).sum(), (m & B).sum(), (m & C).sum()], p[m & ~A].sum()))
rows.sort(key=lambda r: -r[4])
print(f"{'더 사는 밤 (지표 상위20%)':26s} {'앞':>11s} {'뒤':>11s} {'애프터':>11s}  뒤+애프터  세기간+")
for f, th, add, n, s in rows:
    star = "★" if all(a > 0 for a in add) else ""
    print(f"{f:26s} {add[0]:+6.1f}({n[0]:2d}) {add[1]:+6.1f}({n[1]:2d}) {add[2]:+6.1f}({n[2]:2d})  {s:+6.1f}  {star}")
# 상위 조합: 기존 '닛케이·DAX+VIX' OR 지표
print("\n[조합] 닛케이·DAX하락+VIX≥16  OR  지표 상위20%")
base_m = J.nk_dax_vix.values == 1
for f, th, add, n, s in rows[:10]:
    if f.startswith("닛케이"): continue
    mm = base_m | (J[f].values >= th)
    a = [p[mm & A].sum(), p[mm & B].sum(), p[mm & C].sum()]; nn = [(mm & A).sum(), (mm & B).sum(), (mm & C).sum()]
    print(f"  + {f:22s} {a[0]:+6.1f}({nn[0]:2d}) {a[1]:+6.1f}({nn[1]:2d}) {a[2]:+6.1f}({nn[2]:2d})  {'★' if all(x > 0 for x in a) else ''}")
# 문턱 고원 확인 (상위 10/20/30/40%)
print("\n[문턱 고원] 상위 n% 를 더 사는 밤으로 (앞 / 뒤 / 애프터)")
for f in [r[0] for r in rows[:6] if not r[0].startswith("닛케이")]:
    out = []
    for q in (.9, .8, .7, .6):
        th = J[f][A].quantile(q); m = (J[f] >= th).values
        out.append(f"{int((1-q)*100)}%: {p[m & A].sum():+.1f}/{p[m & B].sum():+.1f}/{p[m & C].sum():+.1f}")
    print(f"  {f:22s} " + " | ".join(out))
