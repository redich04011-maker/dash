"""위험회피 — 덜 사기/일찍 팔기 후보 (시간외 단위 근사: 밤 돈 1을 그 밤 판들에 똑같이, 판 손익 = r0800 − 0.25)
L1 밤 일정(CPI·PPI·NFP·FOMC·PCE, 그 밤 22:30/04:00 KST 발표) 밤 쉬기/반
L2 저녁에 오른 판 19:56에 팔기(밤 안 넘김): 판 손익 = eve − 0.25
L3 밤 단위 깃발로 반 매수 (잠잠·과열 등)
L4 한 밤 판 수 상한
지표: 앞/뒤/애프터 합, 최악 밤, −2%↓ 밤 수, 줄인 돈 대비 피한 손실
"""
import numpy as np, pandas as pd
P = pd.read_csv("exchange/from_local/20261104_PLAYS_v1.csv", dtype={"d": str, "c": str}); P["d"] = pd.to_datetime(P.d)
E = pd.read_csv("exchange/from_research/econ_events.csv", dtype={"d": str}); E["d"] = pd.to_datetime(E.d)
F = pd.read_csv("night_features2.csv", parse_dates=["d"]).set_index("d")
big = E[E.종류.isin(["CPI", "PPI", "NFP", "FOMC", "PCE"])].groupby("d").종류.apply(lambda s: ",".join(sorted(set(s))))
P = P.join(big.rename("ev"), on="d").join(F[["vix", "vix_term", "ma20gap", "p_nq", "hrng24", "n225", "dax_gap"]], on="d")
P["aft"] = P.aft.astype(str) == "True"
nights = P.groupby("d").agg(aft=("aft", "first")).sort_index()
pre_n = nights.index[~nights.aft]; mid = pre_n[len(pre_n) // 2]
per = lambda d: "애프터" if d >= pd.Timestamp("2026-09-15") else ("앞" if d < mid else "뒤")
P["per"] = P.d.map(per)
print(f"판 {len(P)} / 밤 {P.d.nunique()} (앞·뒤 경계 {mid.date()}) / 일정 있는 밤 {P[P.ev.notna()].d.nunique()}\n")


def evaluate(w, pnl=None, name=""):
    """w: 판별 비중(1=그대로, 0.5=반, 0=쉼), pnl: 판별 손익(기본 r0800−0.25)"""
    x = P.assign(w=w, g=(P.r0800 - 0.25) if pnl is None else pnl)
    nt = x.groupby("d").apply(lambda s: (s.g * s.w).sum() / len(s), include_groups=False)   # 밤 돈 1 기준
    perN = nt.index.map(per)
    sums = [nt[perN == k].sum() for k in ("앞", "뒤", "애프터")]
    return sums, nt.min(), (nt <= -2).sum(), x.w.sum()


base = evaluate(np.ones(len(P)))
def row(nm, r):
    s, worst, n2, w = r
    d = [s[i] - base[0][i] for i in range(3)]
    print(f"  {nm:44s} {s[0]:+6.1f} / {s[1]:+6.1f} / {s[2]:+6.1f}  (차 {d[0]:+5.1f}/{d[1]:+5.1f}/{d[2]:+5.1f})  최악 {worst:+.2f}  −2%↓밤 {n2:2d}  판 {w:.0f}")
print(f"  {'':44s} {'앞':>6s}   {'뒤':>6s}   {'애프터':>6s}")
row("지금", base)

print("\n[L1] 밤 일정")
for nm, m in (("CPI·PPI·NFP·FOMC·PCE 밤", P.ev.notna()), ("CPI 밤", P.ev.fillna("").str.contains("CPI")),
              ("NFP 밤", P.ev.fillna("").str.contains("NFP")), ("FOMC 밤", P.ev.fillna("").str.contains("FOMC"))):
    s = P[m]
    print(f"  {nm}: {s.d.nunique()}밤 {len(s)}판 | 판당 애프터 전 {(s.r0800[~s.aft]-.25).mean():+.2f} ({(~s.aft).sum()}판) / 애프터 {(s.r0800[s.aft]-.25).mean():+.2f} ({s.aft.sum()}판) | 전체 판당 전 {(P.r0800[~P.aft]-.25).mean():+.2f} 후 {(P.r0800[P.aft]-.25).mean():+.2f}")
    row(f"  → {nm} 쉼", evaluate(np.where(m, 0, 1))); row(f"  → {nm} 반", evaluate(np.where(m, .5, 1)))

print("\n[L2] 저녁에 오른 판 19:56에 팔기 (밤 안 넘김)")
ok = P.eve.notna()
for th in (0.5, 1.0, 2.0, 3.0):
    m = ok & (P.eve >= th)
    pnl = np.where(m, P.eve - 0.25, P.r0800 - 0.25)
    s = P[m]
    print(f"  저녁몫 ≥+{th}%: {len(s)}판, 그 판 밤몫 평균 앞 {s.night[s.per=='앞'].mean():+.2f} / 뒤 {s.night[s.per=='뒤'].mean():+.2f} / 애프터 {s.night[s.per=='애프터'].mean():+.2f} (판 {(s.per=='앞').sum()}/{(s.per=='뒤').sum()}/{(s.per=='애프터').sum()})")
    row(f"  → 19:56 팔기", evaluate(np.ones(len(P)), pnl))

print("\n[L3] 밤 단위 깃발로 반 매수 (문턱 = 앞 기간 분위)")
A = P.per == "앞"
for f, q, side, nm in (("vix_term", .2, "≤", "VIX/VIX3M 하위20%(잠잠)"), ("vix", .2, "≤", "VIX 하위20%"), ("ma20gap", .8, "≥", "20일선 위 상위20%(과열)"),
                       ("p_nq", .8, "≥", "전날 나스닥 급등 상위20%"), ("hrng24", .2, "≤", "직전24h 범위 하위20%(잠잠)"), ("n225", .8, "≥", "닛케이 급등 상위20%")):
    th = P[f][A].quantile(q); m = (P[f] <= th) if side == "≤" else (P[f] >= th)
    s = P[m]
    print(f"  {nm} ({side}{th:.4g}): 판당 앞 {(s.r0800[s.per=='앞']-.25).mean():+.2f} / 뒤 {(s.r0800[s.per=='뒤']-.25).mean():+.2f} / 애프터 {(s.r0800[s.per=='애프터']-.25).mean():+.2f}  (판 {(s.per=='앞').sum()}/{(s.per=='뒤').sum()}/{(s.per=='애프터').sum()})")
    row("  → 반 매수", evaluate(np.where(m, .5, 1)))

print("\n[L4] 한 밤 판 수별 (밤 돈 1을 나눔 → 판 수가 많을수록 분산)")
cnt = P.groupby("d").size()
for k in (1, 2, 3):
    ds = cnt[cnt == k].index; nt = P[P.d.isin(ds)].groupby("d").r0800.mean() - .25
    print(f"  {k}판 밤 {len(ds)}: 밤 평균 {nt.mean():+.2f}, 최악 {nt.min():+.2f}, −2%↓ {(nt<=-2).sum()}")

print("\n[참고] 손실 큰 밤 상위 8 — 어떤 깃발이 켜졌었나")
nt = P.groupby("d").apply(lambda s: (s.r0800 - .25).mean(), include_groups=False).nsmallest(8)
for d, v in nt.items():
    s = P[P.d == d].iloc[0]
    print(f"  {d.date()} 밤 {v:+.2f} | 일정 {s.ev if isinstance(s.ev, str) else '-'} | VIX {s.vix:.1f} VIX/3M {s.vix_term:.3f} 20일선 {s.ma20gap*100:+.1f}% 전날NQ {s.p_nq*100:+.1f}% 24h범위 {s.hrng24*100:.1f}% | 저녁몫 " +
          ", ".join(f"{e:+.1f}" for e in P[P.d == d].eve.dropna()))
