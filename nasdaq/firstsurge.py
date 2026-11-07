"""시간외 덧: '앞 20일 내리거나 쉬던 종목이 산 날 처음 급등' 판 — 애프터에서 왜 뒤집혔나 + '첫 급등 판 반 매수' 판정
종목 일봉: yfinance (코드.KS / 코드.KQ). 판 정보: exchange/from_local/20261104_PLAYS_v1.csv
정의(산 날 d 기준, 모두 d 장 마감 전후에 앎)
  prev20 = 종가(d-1)/종가(d-21) − 1   (전날까지 20일)
  day    = 종가(d)/종가(d-1) − 1      (산 날 등락)
  prev1  = 종가(d-1)/종가(d-2) − 1    (전날 등락)
  fade   = 종가(d)/고가(d) − 1        (산 날 고점 대비 밀림)
  open_jump = 시가(d)/종가(d-1) − 1
"""
import numpy as np, pandas as pd, yfinance as yf
P = pd.read_csv("exchange/from_local/20261104_PLAYS_v1.csv", dtype={"d": str, "d2": str, "c": str}); P["d"] = pd.to_datetime(P.d)
P["aft"] = P.aft.astype(str) == "True"
codes = sorted(P.c.unique())
data = {}
for suf in (".KS", ".KQ"):
    need = [c for c in codes if c not in data]
    if not need: break
    raw = yf.download([c + suf for c in need], start="2025-06-01", end="2026-11-05", interval="1d", auto_adjust=False, progress=False, group_by="ticker", threads=True)
    for c in need:
        try:
            df = raw[c + suf].dropna(how="all")
            if len(df) > 30: data[c] = df
        except Exception: pass
print(f"종목 {len(codes)}개 중 일봉 확보 {len(data)}개")

rows = []
for i, r in P.iterrows():
    df = data.get(r.c)
    if df is None: rows.append({}); continue
    df = df[["Open", "High", "Low", "Close"]].dropna(); df.index = pd.to_datetime(df.index).tz_localize(None)
    if r.d not in df.index: rows.append({}); continue
    k = df.index.get_loc(r.d)
    if k < 22: rows.append({}); continue
    C, O, H = df.Close.values, df.Open.values, df.High.values
    rows.append(dict(prev20=C[k-1]/C[k-21]-1, day=C[k]/C[k-1]-1, prev1=C[k-1]/C[k-2]-1, fade=C[k]/H[k]-1, open_jump=O[k]/C[k-1]-1,
                     intraday=C[k]/O[k]-1, prev5=C[k-1]/C[k-6]-1))
X = pd.concat([P, pd.DataFrame(rows, index=P.index)], axis=1)
X.to_csv("exchange/from_research/20261104_plays_daily.csv", index=False, encoding="utf-8-sig")
X = X.dropna(subset=["day"])
X["g"] = X.r0800 - 0.25
pre, aft = ~X.aft, X.aft
nights = X.d.drop_duplicates().sort_values(); pn = nights[nights < pd.Timestamp("2026-09-15")]; mid = pn.iloc[len(pn)//2]
X["per"] = np.where(X.aft, "애프터", np.where(X.d < mid, "앞", "뒤"))
print(f"분석 판 {len(X)} (애프터 전 {pre.sum()} / 애프터 {aft.sum()}), 앞·뒤 경계 {mid.date()}\n")


def desc(m, nm):
    out = []
    for p_ in ("앞", "뒤", "애프터"):
        s = X[m & (X.per == p_)]
        out.append(f"{p_} {len(s):2d}판 이김 {(s.g>0).mean()*100 if len(s) else 0:3.0f}% 판당 {s.g.mean() if len(s) else np.nan:+.2f} 저녁 {s.eve.mean():+.2f} 밤 {s.night.mean():+.2f}")
    print(f"  {nm:34s} " + " | ".join(out))


print("[1] 꼴별 성과 (판당 = r0800−0.25, 저녁/밤 = 산 값→19:59 / 19:59→08:00)")
desc(np.ones(len(X), bool), "전체")
first = (X.prev20 <= 0) & (X.day >= .08)
desc(first, "첫 급등 (전날까지20일≤0 & 산 날≥+8%)")
desc((X.prev20 <= -.10) & (X.day >= .08), "  └ 전날까지20일≤−10%")
desc((X.prev20 > 0) & (X.day >= .08), "이어 급등 (20일>0 & 산 날≥+8%)")
desc((X.prev1 >= .08) & (X.day.abs() < .03), "전날 급등 → 산 날 쉼(±3%)")
desc(first & (X.fade <= -.08), "첫 급등 + 고점 대비 −8%↓ 밀림")
desc(first & (X.fade > -.08), "첫 급등 + 고점 근처 마감")
desc(first & (X.open_jump >= .05), "첫 급등 + 시가부터 +5%↑ 갭")

print("\n[2] 왜 뒤집혔나 — 첫 급등 판의 저녁·밤 쪼개기 (애프터 전 vs 애프터)")
for nm, m in (("애프터 전", pre & first), ("애프터", aft & first)):
    s = X[m]
    print(f"  {nm}: {len(s)}판 | 산 날 등락 {s.day.mean()*100:+.1f}% 고점대비 {s.fade.mean()*100:+.1f}% 시가갭 {s.open_jump.mean()*100:+.1f}% | 저녁몫 {s.eve.mean():+.2f} 밤몫 {s.night.mean():+.2f} 저녁낙폭 {s['저녁낙폭'].mean():+.2f} | r0800 {s.r0800.mean():+.2f}")
print("  ※ 저녁몫이 크게 −이면 '저녁장 매물'(19:59까지 흘러내림), 밤몫이 −이면 '아침에 못 오름'")
print("  애프터 첫 급등 판 목록:")
for _, r in X[aft & first].sort_values("g").iterrows():
    print(f"    {r.d.date()} {r.c} 산날 {r.day*100:+.1f}% 전날까지20일 {r.prev20*100:+.1f}% 고점대비 {r.fade*100:+.1f}% | 저녁 {r.eve:+.2f} 밤 {r.night:+.2f} 판당 {r.g:+.2f}")

print("\n[3] '첫 급등 판 반 매수' 판정 (밤 돈 1을 판들에 똑같이, 그 판만 0.5)")
def ev(w):
    nt = X.assign(w=w).groupby("d").apply(lambda s: (s.g * s.w).sum() / len(s), include_groups=False)
    per = nt.index.map(lambda d: "애프터" if d >= pd.Timestamp("2026-09-15") else ("앞" if d < mid else "뒤"))
    return [nt[per == k].sum() for k in ("앞", "뒤", "애프터")], nt.min(), (nt <= -2).sum()
b = ev(np.ones(len(X)))
print(f"  지금: {b[0][0]:+.1f}/{b[0][1]:+.1f}/{b[0][2]:+.1f} 최악 {b[1]:+.2f} −2↓밤 {b[2]}")
for p20 in (0, -.05, -.10):
    for dth in (.06, .08, .10, .15):
        m = ((X.prev20 <= p20) & (X.day >= dth)).values
        s = ev(np.where(m, .5, 1))
        print(f"  전날까지20일≤{p20*100:+.0f}% & 산날≥+{dth*100:.0f}% ({m.sum():3d}판, 애프터 {(m & X.aft.values).sum()}): "
              f"차 {s[0][0]-b[0][0]:+5.1f}/{s[0][1]-b[0][1]:+5.1f}/{s[0][2]-b[0][2]:+5.1f}  최악 {s[1]:+.2f} −2↓밤 {s[2]}")
