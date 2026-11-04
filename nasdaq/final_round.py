"""집중 라운드: 우리 밤 '한 판 더 / 덜 사기' 후보 총점검
나눔: 앞(애프터 전 절반) / 뒤 / 애프터(2026-09-15~). 문턱 = 앞 기간 우리 밤 기준 분위(고정)
판정: 초과분(고른 밤 판당 − 기간 판당) 세 기간 모두 같은 부호 + 뒤섞기 p(뒤+애프터, 3000회)
모두 20:00 KST에 아는 값만 사용
"""
import itertools, numpy as np, pandas as pd
rng = np.random.default_rng(11)
F = pd.read_csv("night_features2.csv", parse_dates=["d"]).set_index("d").sort_index()
O = pd.read_csv("in/NIGHTS_시간외_우리성과_밤별.csv", dtype={"d": str}); O["d"] = pd.to_datetime(O.d); O = O.set_index("d").sort_index()

# ---------- 새 지표 ----------
pct = lambda s: s.rolling(250, min_periods=120).apply(lambda x: (x[-1] >= x).mean(), raw=True)   # 1년 내 백분위(그날 포함, 과거만)
for c in ("move", "vxn", "vvix", "hrng24", "range1", "hv24", "vix"):
    F[c + "_p"] = pct(F[c])
F["move_d1"] = F.move.pct_change(); F["move_d5"] = F.move.pct_change(5)
F["vxn_d1"] = F.vxn.pct_change(); F["vvix_d1"] = F.vvix.pct_change()
F["prev_r2"] = F.r2.shift(1)                      # 어젯밤 나스닥 밤(오늘 08:00에 앎)
F["prev_absr2"] = F.r2.abs().shift(1)
F["hr_down"] = F.hrng24 * (F.nq_gap < 0)           # 흔들림 × 지금 내려 있음
F["hr_up"] = F.hrng24 * (F.nq_gap > 0)
J = O.join(F, how="inner")
J["prev_pnl"] = J["우리_08시_판당"].shift(1)        # 우리 직전 판(다음 판 20:00 전에 앎)
J["dow"] = J.index.dayofweek
p = J["우리_08시_판당"].values
AFTER = pd.Timestamp("2026-09-15"); pre = J.index[J.index < AFTER]; mid = pre[len(pre) // 2]
A, B, C = J.index < mid, (J.index >= mid) & (J.index < AFTER), J.index >= AFTER
BC = B | C; idxBC = np.where(BC)[0]
print(f"우리 밤 {len(J)} = 앞 {A.sum()} / 뒤 {B.sum()} / 애프터 {C.sum()} | 기간 판당 {p[A].mean():+.3f} / {p[B].mean():+.3f} / {p[C].mean():+.3f}\n")


def score(m):
    e = [p[m & P].mean() - p[P].mean() if (m & P).sum() else np.nan for P in (A, B, C)]
    n = [(m & P).sum() for P in (A, B, C)]; s = [p[m & P].sum() for P in (A, B, C)]
    k = (m & BC).sum()
    if k == 0: return e, n, s, np.nan, np.nan
    obs = p[m & BC].mean() - p[BC].mean()
    sims = np.array([p[rng.choice(idxBC, k, replace=False)].mean() - p[BC].mean() for _ in range(3000)])
    return e, n, s, np.mean(sims >= obs), np.mean(sims <= obs)


def rule(f, q=.8, hi=True):
    x = J[f]; th = x[A].quantile(q)
    return ((x >= th) if hi else (x <= th)).fillna(False).values, th


def show(title, items, side="up"):
    print(f"[{title}]")
    print(f"  {'규칙':34s} {'초과 앞(밤)':>11s} {'뒤(밤)':>11s} {'애프터(밤)':>11s}   p")
    out = []
    for nm, m in items:
        e, n, s, pu, pd_ = score(m); pv = pu if side == "up" else pd_
        ok = all((x > 0 if side == "up" else x < 0) for x in e if not np.isnan(x)) and not any(np.isnan(e)) and pv < .05
        out.append((nm, e, n, s, pv, ok))
    out.sort(key=lambda r: r[4] if not np.isnan(r[4]) else 9)
    for nm, e, n, s, pv, ok in out:
        f = lambda v, k: f"{v:+.2f}({k:3d})" if not np.isnan(v) else f"   -({k:3d})"
        print(f"  {nm:34s} {f(e[0], n[0])} {f(e[1], n[1])} {f(e[2], n[2])}  {pv:.3f}{' ★' if ok else ''}")
    print()
    return out


# ---------- 1) 백분위·변화량·방향·어젯밤·우리직전·요일 (한 판 더) ----------
ND = ((J.n225 < 0) & (J.dax_gap < 0) & (J.vix >= 16)).values
H24, th_h24 = rule("hrng24")
items = [("기존: 닛케이·DAX↓+VIX≥16", ND), ("기존: 직전24h 범위 상위20%", H24),
         ("기존 최선: 둘 중 하나", ND | H24)]
for f, lab in [("move_p", "MOVE 1년백분위"), ("vxn_p", "VXN 1년백분위"), ("vvix_p", "VVIX 1년백분위"), ("vix_p", "VIX 1년백분위"),
               ("hrng24_p", "직전24h범위 1년백분위"), ("range1_p", "전일고저폭 1년백분위"), ("hv24_p", "직전24h변동성 1년백분위"),
               ("move_d1", "MOVE 1일 상승"), ("move_d5", "MOVE 5일 상승"), ("vxn_d1", "VXN 1일 상승"), ("vvix_d1", "VVIX 1일 상승"),
               ("prev_absr2", "어젯밤 나스닥 출렁임 큼"), ("hr_down", "흔들림×지금 하락"), ("hr_up", "흔들림×지금 상승")]:
    m, th = rule(f); items.append((f"{lab} 상위20%", m))
m, _ = rule("prev_r2", .2, hi=False); items.append(("어젯밤 나스닥 밤 하위20%(급락)", m))
m, _ = rule("prev_r2", .8); items.append(("어젯밤 나스닥 밤 상위20%(급등)", m))
items.append(("우리 직전 판 손실", (J.prev_pnl < 0).values)); items.append(("우리 직전 판 −1% 이하", (J.prev_pnl <= -1).values))
for d_, nm in enumerate("월화수목금"): items.append((f"{nm}요일", (J.dow == d_).values))
up = show("1) 한 판 더 — 새 후보", items, "up")

# ---------- 2) 덜 사는 밤 ----------
items = []
for f, lab in [("hrng24", "직전24h 범위"), ("range1", "전일 고저폭"), ("vix", "VIX"), ("vix_p", "VIX 1년백분위"), ("move_p", "MOVE 1년백분위"),
               ("hv24", "직전24h 변동성"), ("vix_term", "VIX/VIX3M"), ("neg_ma20", "20일선 아래폭")]:
    m, _ = rule(f, .2, hi=False); items.append((f"{lab} 하위20%(잠잠)", m))
m, _ = rule("ma20gap", .8); items.append(("20일선 위 상위20%(과열)", m))
m, _ = rule("p_nq", .8); items.append(("전날 나스닥 상승 상위20%", m))
m, _ = rule("n225", .8); items.append(("닛케이 상승 상위20%", m))
calm = (J.hrng24 <= J.hrng24[A].quantile(.2)) & (J.vix <= J.vix[A].quantile(.3))
items.append(("잠잠 조합: 24h범위 하위20% & VIX 하위30%", calm.values))
dn = show("2) 덜 사는 밤 (초과분이 세 기간 모두 −)", items, "down")

# ---------- 3) 조합 (OR) 상위 ----------
good = [(nm, m) for nm, m in [(r[0], dict(itm for itm in [(i[0], i[1]) for i in items]).get(r[0])) for r in []]]
cand = {"닛케이·DAX↓+VIX≥16": ND, "직전24h범위 상위20%": H24}
for nm, e, n, s, pv, ok in up:
    if ok and nm not in ("기존: 닛케이·DAX↓+VIX≥16", "기존: 직전24h 범위 상위20%", "기존 최선: 둘 중 하나"):
        cand[nm] = dict([(i[0], i[1]) for i in [*[(x[0], x[1]) for x in []]]]).get(nm)
# 재구성(통과 후보 마스크)
allm = {}
for f, lab in [("move_p", "MOVE 1년백분위"), ("vxn_p", "VXN 1년백분위"), ("vvix_p", "VVIX 1년백분위"), ("vix_p", "VIX 1년백분위"),
               ("hrng24_p", "직전24h범위 1년백분위"), ("range1_p", "전일고저폭 1년백분위"), ("hv24_p", "직전24h변동성 1년백분위"),
               ("move_d1", "MOVE 1일 상승"), ("move_d5", "MOVE 5일 상승"), ("vxn_d1", "VXN 1일 상승"), ("vvix_d1", "VVIX 1일 상승"),
               ("prev_absr2", "어젯밤 나스닥 출렁임 큼"), ("hr_down", "흔들림×지금 하락"), ("hr_up", "흔들림×지금 상승"), ("range1", "전일고저폭")]:
    allm[f"{lab} 상위20%"] = rule(f)[0]
allm["VIX 전일 상승 상위20%"] = rule("vix_d1")[0]
passed = [r[0] for r in up if r[5] and r[0] in allm]
pool = {"닛케이·DAX↓+VIX≥16": ND, "직전24h범위 상위20%": H24, **{k: allm[k] for k in passed}}
print(f"[3] 조합 (OR) — 재료: {list(pool)}")
res = []
for r in (2, 3):
    for combo in itertools.combinations(pool, r):
        m = np.zeros(len(J), bool)
        for k in combo: m |= pool[k]
        e, n, s, pu, _ = score(m)
        if min(e) > 0 and pu < .02: res.append((" | ".join(combo), e, n, s, pu))
res.sort(key=lambda r: (-sum(r[3][1:]), r[4]))
for nm, e, n, s, pu in res[:10]:
    print(f"  {nm:70s} 이익 {s[0]:+.1f}({n[0]})/{s[1]:+.1f}({n[1]})/{s[2]:+.1f}({n[2]})  초과 {e[0]:+.2f}/{e[1]:+.2f}/{e[2]:+.2f} p={pu:.3f}")
print()

# ---------- 4) 최선 조합 문턱 고원 ----------
print("[4] 고원: 닛케이·DAX↓+VIX≥k  OR  직전24h범위 상위 q  (이익 앞/뒤/애프터, p)")
for vk in (14, 16, 18, 20):
    row = []
    for q in (.9, .8, .7, .6):
        m = ((J.n225 < 0) & (J.dax_gap < 0) & (J.vix >= vk)).values | rule("hrng24", q)[0]
        e, n, s, pu, _ = score(m)
        row.append(f"상위{int(round((1-q)*100))}%: {s[0]:+.0f}/{s[1]:+.0f}/{s[2]:+.0f} p{pu:.2f}")
    print(f"  VIX≥{vk}: " + " | ".join(row))
print()

# ---------- 5) 더 사기 + 덜 사기 동시 ----------
best_up = ND | H24
calm_m = [r for r in dn if r[5]]
print("[5] 판 크기 시나리오 (합 앞/뒤/애프터, 최악의 밤, 밤당 표준편차)")
w0 = np.ones(len(J))
scen = {"지금(늘 1판)": w0, "더사기(최선 조합 밤 2판)": np.where(best_up, 2, 1)}
for nm, e, n, s, pv, ok in calm_m[:2]:
    mm = dict(items)[nm]
    scen[f"덜사기({nm} 0판)"] = np.where(mm, 0, 1)
    scen[f"둘 다(최선 2판, {nm} 0판)"] = np.select([best_up, mm], [2, 0], 1)
for nm, w in scen.items():
    r = p * w
    print(f"  {nm:52s} {r[A].sum():+6.1f} / {r[B].sum():+6.1f} / {r[C].sum():+6.1f}  최악 {r.min():+.2f}  표준편차 {r.std():.2f}  투입판 {w.sum():.0f}")

# ---------- 저장 ----------
Fx = F.copy()
fl = pd.DataFrame(index=Fx.index)
fl["닛케이DAX하락_VIX16"] = ((Fx.n225 < 0) & (Fx.dax_gap < 0) & (Fx.vix >= 16)).astype(int)
fl["직전24h범위_상위"] = (Fx.hrng24 >= th_h24).astype(int)
fl["최선조합_한판더"] = ((fl.iloc[:, 0] == 1) | (fl.iloc[:, 1] == 1)).astype(int)
for k in passed:
    f = {v: k_ for k_, v in []}.get(k)
fl["직전24h범위_%"] = (Fx.hrng24 * 100).round(2); fl["VIX"] = Fx.vix.round(2); fl["닛케이"] = (Fx.n225 * 100).round(2); fl["DAX갭"] = (Fx.dax_gap * 100).round(2)
fl["MOVE_1년백분위"] = Fx.move_p.round(2)
fl.index = fl.index.strftime("%Y%m%d"); fl.index.name = "d"
fl.to_csv("night_addone_flags_v2.csv", encoding="utf-8-sig")
print(f"\n저장: night_addone_flags_v2.csv  (직전24h범위 문턱 {th_h24*100:.2f}%)")
