"""vol5 보정: 우리 밤은 평균이 +라 아무 밤이나 더 사도 +가 나옴 → '고른 밤 평균 − 그 기간 평균'(초과분)과 뒤섞기 p로 다시 판정"""
import io, contextlib, numpy as np, pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    import vol5 as v
J, p, A, B, C = v.J, v.p.values, v.A, v.B, v.C
rng = np.random.default_rng(3)
def ex(m, P):  # 초과분(밤당) 
    return p[m & P].mean() - p[P].mean() if (m & P).sum() else np.nan
print(f"기간 평균 판당: 앞 {p[A].mean():+.3f} / 뒤 {p[B].mean():+.3f} / 애프터 {p[C].mean():+.3f}\n")
print(f"{'더 사는 밤':24s} {'앞 초과':>12s} {'뒤 초과':>12s} {'애프터 초과':>12s}  뒤섞기p(뒤+애프터)")
res = []
cands = [r[0] for r in v.rows]
for f in cands:
    if f.startswith("닛케이"): m = J.nk_dax_vix.values == 1
    else: th = J[f][A].quantile(.8); m = (J[f] >= th).values
    e = [ex(m, A), ex(m, B), ex(m, C)]; n = [(m & A).sum(), (m & B).sum(), (m & C).sum()]
    # 뒤+애프터에서 같은 수 무작위 밤을 골랐을 때 초과분 합이 이 이상일 확률
    BC = B | C; k = (m & BC).sum(); obs = p[m & BC].sum() - k * p[BC].mean()
    idx = np.where(BC)[0]
    sims = [p[rng.choice(idx, k, replace=False)].sum() - k * p[BC].mean() for _ in range(3000)] if k else [0]
    pv = np.mean(np.array(sims) >= obs) if k else np.nan
    res.append((f, e, n, pv))
res.sort(key=lambda r: r[3])
for f, e, n, pv in res:
    star = "★" if all((x > 0) for x in e if not np.isnan(x)) and min(n) >= 3 and pv < .1 else ""
    print(f"{f:24s} {e[0]:+.2f}({n[0]:2d}) {e[1]:+.2f}({n[1]:2d}) {e[2]:+.2f}({n[2]:2d})   p={pv:.3f} {star}")
