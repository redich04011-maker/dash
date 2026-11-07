"""쉬고 난 다음날(월요일·휴일 다음날) 선물 갭 가중치 연구
갭 = 주말점프(전 거래일 16:00 ET → 선물 재개장 첫 1시간) + 재개장 후 흐름(→ 당일 20:00 KST)"""
import numpy as np, pandas as pd
from pattern import df, gap, nq, asof, ET
from sklearn.linear_model import LogisticRegression

d = df[df.y.notna()].copy(); d["gap"] = gap[d.index]; d["y"] = d.y.astype(bool)
dates = list(d.date)
prev = [None] + dates[:-1]
d["brk"] = [(p is not None and (cur - p).days > 1 and not (cur.dayofweek == 0 and (cur - p).days == 3)) or cur.dayofweek == 0
            for p, cur in zip(prev, dates)]
d["holiday_after"] = [(p is not None and (cur - p).days > 1 and cur.dayofweek != 0) or (cur.dayofweek == 0 and p is not None and (cur - p).days > 3)
                      for p, cur in zip(prev, dates)]
jump, drift = [], []
for k, r in d.iterrows():
    dt = r.date; p = prev[dates.index(dt)]
    if not r.brk or p is None: jump.append(np.nan); drift.append(np.nan); continue
    c0 = pd.Timestamp(f"{p.date()} 16:00", tz=ET).tz_convert("UTC"); base = asof(nq, c0)
    after = nq[nq.index > c0 + pd.Timedelta(hours=20)]          # 주말/휴장 이후 첫 봉
    if not len(after): jump.append(np.nan); drift.append(np.nan); continue
    j = after.iloc[0] / base - 1; jump.append(j); drift.append((1 + r.gap) / (1 + j) - 1)
d["jump"], d["drift"] = jump, drift
br = d[d.brk & d.jump.notna()]; rg = d[~d.brk]
acc = lambda s, y: ((s > 0) == y).mean() * 100
print(f"쉬고 난 다음날 {len(br)}일 (그중 휴일 다음날 {int(br.holiday_after.sum())}일) / 일반일 {len(rg)}일\n")
print("[적중률] 각 신호 방향 = 당일 종가 방향")
print(f"  일반일   8시 갭 전체            {acc(rg.gap, rg.y):5.1f}%")
print(f"  쉰다음날 8시 갭 전체            {acc(br.gap, br.y):5.1f}%")
print(f"  쉰다음날 주말점프만              {acc(br.jump, br.y):5.1f}%")
print(f"  쉰다음날 재개장 후 흐름만         {acc(br.drift, br.y):5.1f}%")
for lo in (0.0025, 0.005):
    m = br.drift.abs() >= lo
    print(f"  쉰다음날 재개장후흐름 |≥{lo*100:.2f}%| {acc(br.drift[m], br.y[m]):5.1f}% ({m.sum()}일)")
print("\n[가중치 탐색] 신호 = w × 주말점프 + 재개장후흐름")
for w in (-1, -0.5, 0, 0.25, 0.5, 0.75, 1, 1.5):
    s = w * br.jump + br.drift
    print(f"  w={w:+.2f}: {acc(s, br.y):5.1f}%")
# walk-forward 로지스틱으로 가중치 학습
X = br[["jump", "drift"]].values; y = br.y.values; P = np.full(len(y), np.nan)
for s in range(30, len(y), 5):
    m = LogisticRegression(C=1.0).fit(X[:s], y[:s]); P[s:s + 5] = m.predict_proba(X[s:s + 5])[:, 1]
k = ~np.isnan(P)
print(f"\n  학습형(walk-forward, {k.sum()}일): {((P[k] > .5) == y[k]).mean()*100:.1f}%  | 같은 기간 갭전체 {acc(br.gap.values[k], y[k]):.1f}%")
m = LogisticRegression(C=1.0).fit(X, y)
print(f"  전체 학습 계수: 주말점프 {m.coef_[0][0]:+.1f}  재개장후흐름 {m.coef_[0][1]:+.1f}  (비율 w≈{m.coef_[0][0]/m.coef_[0][1]:+.2f})")
print("\n[주말점프 크기별] 큰 점프는 되돌리나?")
for lo, hi in ((0, .005), (.005, .01), (.01, 1)):
    m = (br.jump.abs() >= lo) & (br.jump.abs() < hi)
    if m.sum(): print(f"  |점프| {lo*100:.1f}~{min(hi*100,99):.1f}%: 점프방향 적중 {acc(br.jump[m], br.y[m]):5.1f}% ({m.sum()}일), 점프 중 장마감까지 남은 비율 {np.median((br.post[m].apply(lambda a: a[-1]) / br.jump[m])):.2f}")
print("\n[최근 쉰다음날 10일]")
for _, r in br.tail(10).iterrows():
    print(f"  {r.date.date()} 점프 {r.jump*100:+.2f}% 흐름 {r.drift*100:+.2f}% 갭 {r.gap*100:+.2f}% → 종가 {r.post[-1]*100:+.2f}%  "
          f"갭{'O' if (r.gap>0)==r.y else 'X'} 흐름{'O' if (r.drift>0)==r.y else 'X'}")
