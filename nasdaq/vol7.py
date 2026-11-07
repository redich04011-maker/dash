"""후보 조합 검증 (초과분 + 뒤섞기). 문턱은 모두 앞 기간 상위20%."""
import io, contextlib, itertools, numpy as np, pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    import vol5 as v
J, p, A, B, C = v.J, v.p.values, v.A, v.B, v.C
rng = np.random.default_rng(5)
def mask(f):
    return (J.nk_dax_vix.values == 1) if f == "ND" else (J[f] >= J[f][A].quantile(.8)).values
names = {"ND": "닛케이·DAX↓+VIX≥16", "hrng24": "직전24h 나스닥 범위", "range1": "전일 나스닥 고저폭", "vix_d1": "VIX 전일 상승", "move": "MOVE(채권변동성)"}
def score(m):
    e = [p[m & P].mean() - p[P].mean() for P in (A, B, C)]; n = [(m & P).sum() for P in (A, B, C)]
    s = [p[m & P].sum() for P in (A, B, C)]
    BC = B | C; k = (m & BC).sum(); obs = p[m & BC].sum() - k * p[BC].mean(); idx = np.where(BC)[0]
    pv = np.mean([p[rng.choice(idx, k, replace=False)].sum() - k * p[BC].mean() >= obs for _ in range(3000)])
    return e, n, s, pv
print(f"{'조합 (OR)':48s} {'한판더 이익 앞/뒤/애프터':26s} {'초과 앞/뒤/애프터':24s} p")
for r in (1, 2, 3):
    for combo in itertools.combinations(names, r):
        m = np.zeros(len(J), bool)
        for f in combo: m |= mask(f)
        e, n, s, pv = score(m)
        if r > 1 and pv > .05: continue
        nm = " | ".join(names[f] for f in combo)
        print(f"{nm:48s} {s[0]:+5.1f}({n[0]:2d}) {s[1]:+5.1f}({n[1]:2d}) {s[2]:+5.1f}({n[2]:2d})  {e[0]:+.2f} {e[1]:+.2f} {e[2]:+.2f}  {pv:.3f}{' ★' if min(e) > 0 else ''}")
