"""시간외 답1 부탁 1~3
1) 애프터에서 '나스닥 밤 ↔ 우리 판' 관계가 약해진 이유: 표본 / 계절·국면 / 국내 저녁장 선반영
2) 우리 판 크게 진 밤(판당 ≤ -2%)을 20:00 정보로 미리 잡기 — 잡은/놓친/헛잡은 수
3) 애프터만으로 '덜 살 밤' 후보 (애프터 판당 − , 애프터 전 같은 부호)
"""
import numpy as np, pandas as pd
rng = np.random.default_rng(21)
F = pd.read_csv("night_features2.csv", parse_dates=["d"]).set_index("d").sort_index()
F["prev_r2"] = F.r2.shift(1); F["prev_abs"] = F.r2.abs().shift(1)
O = pd.read_csv("in/NIGHTS_시간외_우리성과_밤별.csv", dtype={"d": str}); O["d"] = pd.to_datetime(O.d); O = O.set_index("d").sort_index()
J = O.join(F, how="inner"); p = J["우리_08시_판당"]
AFT = pd.Timestamp("2026-09-15"); pre, aft = J.index < AFT, J.index >= AFT
print(f"우리 밤 {len(J)} (애프터 전 {pre.sum()} / 애프터 {aft.sum()})\n")

# ================= 1) 관계 변화 =================
def rel(m):
    s = J[m]
    return dict(n=len(s), corr=np.corrcoef(s["우리_08시_판당"], s.r2)[0, 1], dn=s["우리_08시_판당"][s.r2 <= -.003].mean(),
                ndn=(s.r2 <= -.003).sum(), mean=s["우리_08시_판당"].mean(), c_gap=np.corrcoef(s["우리_08시_판당"], s.nq_gap)[0, 1])
a, b = rel(pre), rel(aft)
print("[1] 우리 판당 ↔ 밤 NQ(20→08) 관계")
print(f"  애프터 전: 상관 {a['corr']:+.2f}, NQ −0.3%↓ 밤({a['ndn']}) 판당 {a['dn']:+.2f}, 평균 {a['mean']:+.2f}, 8시갭과 상관 {a['c_gap']:+.2f}")
print(f"  애프터  : 상관 {b['corr']:+.2f}, NQ −0.3%↓ 밤({b['ndn']}) 판당 {b['dn']:+.2f}, 평균 {b['mean']:+.2f}, 8시갭과 상관 {b['c_gap']:+.2f}")
# 1-a 표본: 애프터 전에서 연속 25밤 창을 모두 뽑아 분포
idx = np.where(pre)[0]; k = aft.sum()
wins = [J.iloc[i:i + k] for i in range(0, len(idx) - k + 1)]
cs = np.array([np.corrcoef(w["우리_08시_판당"], w.r2)[0, 1] for w in wins])
dns = np.array([w["우리_08시_판당"][w.r2 <= -.003].mean() for w in wins])
print(f"\n  1-a 표본 가설: 애프터 전 연속 {k}밤 창 {len(wins)}개")
print(f"      상관 분포 중앙 {np.median(cs):+.2f}, 하위5% {np.quantile(cs,.05):+.2f}, 최저 {cs.min():+.2f} → 애프터 {b['corr']:+.2f} 이하인 창 {np.mean(cs <= b['corr'])*100:.0f}%")
print(f"      NQ하락밤 판당 분포 중앙 {np.nanmedian(dns):+.2f}, 상위5% {np.nanquantile(dns,.95):+.2f}, 최고 {np.nanmax(dns):+.2f} → 애프터 {b['dn']:+.2f} 이상인 창 {np.nanmean(dns >= b['dn'])*100:.0f}%")
# 1-b 계절: 작년 같은 철 (애프터 전 중 9/15~11/02)
season = pre & (((J.index.month == 9) & (J.index.day >= 15)) | (J.index.month == 10) | ((J.index.month == 11) & (J.index.day <= 2)))
c = rel(season)
print(f"\n  1-b 계절 가설: 작년 같은 철(2025-09-15~11-02) {c['n']}밤: 상관 {c['corr']:+.2f}, NQ하락밤({c['ndn']}) 판당 {c['dn']:+.2f}, 평균 {c['mean']:+.2f}")
# 1-c 국면: 애프터와 지표가 비슷한 애프터 전 밤 (VIX, 20일선 이격, 20일 변동성, 24h범위)
cols = ["vix", "ma20gap", "vol20", "hrng24", "vix_term"]
Z = (J[cols] - J[cols][pre].mean()) / J[cols][pre].std()
cen = Z[aft].mean(); dist = ((Z[pre] - cen) ** 2).sum(1)
near = dist.nsmallest(40).index; mm = J.index.isin(near)
c = rel(mm)
print(f"  1-c 국면 가설: 애프터와 지표가 가장 비슷한 애프터 전 40밤: 상관 {c['corr']:+.2f}, NQ하락밤({c['ndn']}) 판당 {c['dn']:+.2f}, 평균 {c['mean']:+.2f}")
print(f"      애프터 평균 지표: " + ", ".join(f"{x} {J[x][aft].mean():.4g}(전 {J[x][pre].mean():.4g})" for x in cols))
# 1-d 국내 저녁장 선반영: 8시까지 NQ(갭)와 우리 판 상관이 애프터에서 커졌나 / 갭 통제 후 밤 NQ 상관
from numpy.linalg import lstsq
for nm, m in (("애프터 전", pre), ("애프터", aft)):
    s = J[m]; X = np.c_[np.ones(len(s)), s.nq_gap * 100, s.r2 * 100]; coef = lstsq(X, s["우리_08시_판당"].values, rcond=None)[0]
    print(f"  1-d {nm}: 판당 = {coef[0]:+.2f} + {coef[1]:+.2f}×8시갭 + {coef[2]:+.2f}×밤NQ   (갭 상관 {np.corrcoef(s['우리_08시_판당'], s.nq_gap)[0,1]:+.2f})")

# ================= 2) 크게 진 밤 미리 잡기 =================
big = (p <= -2).values
print(f"\n[2] 우리 판당 ≤ −2% 밤: 애프터 전 {big[pre].sum()} / 애프터 {big[aft].sum()} (전체 {big.sum()}/{len(J)})")
feats = {"vix": "VIX", "vix_term": "VIX/VIX3M", "vix9_term": "VIX9D/VIX", "ma20gap": "20일선 이격", "neg_ma20": "20일선 아래폭",
         "p_nq": "전날 나스닥", "nq_gap": "8시 갭", "nq_19_20": "19~20시 NQ", "QQQ_chg": "QQQ 프리", "SMH_chg": "SMH 프리", "SOXL_chg": "SOXL 프리",
         "hrng24": "직전24h 범위", "range1": "전일 고저폭", "vix_d1": "VIX 1일 변화", "n225": "닛케이", "dax_gap": "DAX", "kq_ret": "코스닥 당일",
         "kq_rng": "코스닥 고저폭", "BTC_chg": "BTC", "DXY_chg": "달러", "KRW_chg": "원달러", "ZN_chg": "미국채선물", "OIL_chg": "유가",
         "prev_r2": "어젯밤 NQ", "prev_abs": "어젯밤 NQ 출렁임", "vol20": "20일 변동성", "move": "MOVE", "vxn": "VXN"}
res = []
for f, nm in feats.items():
    if f not in J: continue
    x = J[f]
    for q, side in ((.1, "≤"), (.2, "≤"), (.8, "≥"), (.9, "≥")):
        th = x[pre].quantile(q); m = ((x <= th) if side == "≤" else (x >= th)).fillna(False).values
        hit = (m & big)[pre].sum(); fa = (m & ~big)[pre].sum()
        prec = hit / max(m[pre].sum(), 1); rec = hit / max(big[pre].sum(), 1)
        lift = prec / big[pre].mean()
        res.append((f"{nm} {side}{th:+.4g}", m, hit, fa, prec, rec, lift))
res.sort(key=lambda r: -r[6])
print(f"  기준: 아무 밤이나 고르면 큰손실 밤일 확률 {big[pre].mean()*100:.1f}% (애프터 전)")
print(f"  {'규칙(문턱=애프터 전 분위)':28s} {'잡음':>4s} {'헛잡음':>5s} {'정밀도':>6s} {'재현율':>6s} {'배율':>5s} | 애프터: 잡음/헛잡음/놓침 | 쉬면: 피한 손실·버린 이익(애프터 전)")
for nm, m, hit, fa, prec, rec, lift in res[:14]:
    ha, fa_a, miss_a = (m & big)[aft].sum(), (m & ~big)[aft].sum(), (~m & big)[aft].sum()
    lost = -p.values[m & big & pre].sum(); gave = p.values[m & ~big & pre].sum()
    print(f"  {nm:28s} {hit:4d} {fa:5d} {prec*100:5.1f}% {rec*100:5.1f}% {lift:4.1f}배 | {ha}/{fa_a}/{miss_a} | 피함 {lost:+.1f} 버림 {gave:+.1f} 순 {lost-gave:+.1f}")

# 조합: 상위 규칙 2개 AND
print("\n  2개 AND 조합 (정밀도 높이기, 애프터 전 잡음≥4)")
top = res[:12]; comb = []
for i in range(len(top)):
    for j in range(i + 1, len(top)):
        m = top[i][1] & top[j][1]; hit = (m & big)[pre].sum(); n = m[pre].sum()
        if hit >= 4: comb.append((top[i][0] + " & " + top[j][0], m, hit, n - hit, hit / n))
comb.sort(key=lambda r: -r[4])
for nm, m, hit, fa, prec in comb[:8]:
    lost = -p.values[m & big & pre].sum(); gave = p.values[m & ~big & pre].sum()
    print(f"  {nm:60s} 잡음 {hit} 헛잡음 {fa} 정밀도 {prec*100:.0f}% | 애프터 {(m&big)[aft].sum()}/{(m&~big)[aft].sum()} | 순 {lost-gave:+.1f}")

# ================= 3) 애프터만으로 덜 살 밤 =================
print(f"\n[3] 애프터 {aft.sum()}밤에서 '덜 살 밤' (애프터 판당 −, 애프터 전 같은 부호, 애프터 해당 ≥4밤)")
out = []
for f, nm in feats.items():
    if f not in J: continue
    x = J[f]
    for q, side in ((.3, "≤"), (.5, "≤"), (.5, "≥"), (.7, "≥")):
        th = x[aft].quantile(q); m = ((x <= th) if side == "≤" else (x >= th)).fillna(False).values
        if m[aft].sum() < 4: continue
        ea = p.values[m & aft].mean() - p.values[aft].mean(); ep = p.values[m & pre].mean() - p.values[pre].mean()
        if p.values[m & aft].mean() < 0 and ep < 0:
            out.append((f"{nm} {side}{th:+.4g}", m[aft].sum(), p.values[m & aft].mean(), ea, m[pre].sum(), ep, p.values[m & aft].min()))
out.sort(key=lambda r: r[2])
print(f"  애프터 기간 판당 평균 {p.values[aft].mean():+.2f} / 애프터 전 {p.values[pre].mean():+.2f}")
for nm, na, ma, ea, npre, ep, worst in out[:12]:
    print(f"  {nm:28s} 애프터 {na}밤 판당 {ma:+.2f} (초과 {ea:+.2f}, 최악 {worst:+.2f}) | 애프터 전 {npre}밤 초과 {ep:+.2f}")
if not out: print("  해당 없음")
print("\n  애프터 판당 음수 밤 목록: " + ", ".join(f"{d.strftime('%m/%d')}({v:+.2f})" for d, v in p[aft][p[aft] < 0].items()))
