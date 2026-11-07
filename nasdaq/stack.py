"""두 큰 신호(밤8시 갭, 패턴유사도)를 '버리지 않고' 섞는 메타모델
- 입력: 갭 크기, 패턴 상승확률(2종), 추세, 그리고 서로의 곱(상호작용)
- 엇갈리는 날에도 '어느 쪽을 얼마나 믿을지'를 과거 데이터로 학습
- walk-forward: 매 시점 과거만으로 학습
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pattern import df, gap, predict, full

pool = np.array([i for i in range(len(df)) if i >= 120])
pp1 = np.array([predict(i, 20, "모양+크기", False, 1.0)[0] for i in pool])   # 모양+크기+상황
pp2 = np.array([predict(i, 40, "모양만", False, 0.0)[0] for i in pool])      # 모양만
g = gap[pool]; r = df.reg.values[pool].astype(float)
F = pd.DataFrame({
    "gap": g, "p1": pp1 - .5, "p2": pp2 - .5, "reg": r,
    "gap_reg": g * r, "p1_reg": (pp1 - .5) * r,
    "gap_p1": g * (pp1 - .5),                       # 같은 방향이면 +, 엇갈리면 -
    "absgap_p1": np.abs(g) * (pp1 - .5),            # 갭 작을 때 패턴 비중 조절
    "vol": df.vol20.values[pool], "vix": df.vix.values[pool], "ma20": df.ma20gap.values[pool],
})
y = df.y.values[pool]
done = ~pd.isna(y); yb = np.where(done, y, 0).astype(bool)
X = F.values; P = np.full(len(pool), np.nan)
idx = np.where(done)[0]
for s in range(150, len(pool), 20):
    tr = idx[idx < s]
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)).fit(X[tr], yb[tr])
    P[s:s + 20] = m.predict_proba(X[s:s + 20])[:, 1]
k = ~np.isnan(P) & done
hit = (P > .5) == yb; conf = np.abs(P - .5) + .5
gh = (g > 0) == yb; ph = (pp1 > .5) == yb
agree = (g > 0) == (pp1 > .5)
print(f"메타모델 검증 {k.sum()}일\n")
print(f"  갭 방향만         {gh[k].mean()*100:5.1f}%")
print(f"  패턴만            {ph[k].mean()*100:5.1f}%")
print(f"  합의만 따르기      {gh[k & agree].mean()*100:5.1f}% ({(k & agree).sum()}일, 나머지 버림)")
print(f"  ★ 메타모델(전부)   {hit[k].mean()*100:5.1f}% ({k.sum()}일, 버리는 날 없음)\n")
print("  [엇갈린 날] 누가 맞았나")
dk = k & ~agree
print(f"    {dk.sum()}일: 갭쪽 {gh[dk].mean()*100:.1f}% | 패턴쪽 {ph[dk].mean()*100:.1f}% | 메타 {hit[dk].mean()*100:.1f}%")
for lo, hi, nm in ((0, .0025, "갭<0.25%"), (.0025, .005, "갭0.25~0.5%"), (.005, 1, "갭≥0.5%")):
    for strong, sn in ((False, "패턴약"), (True, "패턴강")):
        m = dk & (np.abs(g) >= lo) & (np.abs(g) < hi) & ((np.abs(pp1 - .5) >= .2) == strong)
        if m.sum() >= 3:
            print(f"    {nm:10s} {sn}: {m.sum():3d}일  갭쪽 {gh[m].mean()*100:5.1f}%  패턴쪽 {ph[m].mean()*100:5.1f}%  메타 {hit[m].mean()*100:5.1f}%")
print("\n  [메타 확신구간별]")
for a, b in ((.9, 1.01), (.8, .9), (.7, .8), (.6, .7), (.5, .6)):
    m = k & (conf >= a) & (conf < b)
    if m.sum(): print(f"    {int(a*100)}~{min(int(b*100),100)}%: {hit[m].mean()*100:5.1f}% ({m.sum()}일)")
for th in (.7, .8):
    m = k & (conf >= th); print(f"    확신≥{int(th*100)}% 누적: {hit[m].mean()*100:.1f}% ({m.sum()}일)")
print("\n  [최근 10일]")
last = [j for j in np.where(~np.isnan(P))[0]][-11:]
for j in last:
    res = ("O" if hit[j] else "X") if done[j] else "진행중"
    print(f"    {df.date[pool[j]].date()} 갭 {g[j]*100:+.2f}% 패턴 {pp1[j]*100:3.0f}% → 메타 상승 {P[j]*100:3.0f}% {'상' if P[j]>.5 else '하'} {res}")
m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)).fit(X[idx], yb[idx])
print("\n  [학습된 비중(표준화 계수)]  " + "  ".join(f"{c} {v:+.2f}" for c, v in zip(F.columns, m[-1].coef_[0])))
