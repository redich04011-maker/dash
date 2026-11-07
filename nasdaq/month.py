"""최근 한 달: 밤8시 정보만으로 상/하 + 등급(강/중/약) 예측 → 실제 비교
방향 = 8시 NQ갭 부호 / 등급:
  강 = (갭·패턴 일치 AND |갭|≥0.5%) OR (상승세 AND |갭|≥0.75%)
  약 = 갭·패턴 엇갈림
  중 = 그 외
패턴 = 모양+크기+상황 K=20 (그날 이전 날만 후보)"""
import numpy as np
from pattern import df, gap, predict

nm = {1: "상승세", 0: "횡보세", -1: "하락세"}
rows = list(range(len(df) - 22, len(df)))
tot = {"강": [0, 0], "중": [0, 0], "약": [0, 0]}
print(f"{'날짜':10s} {'추세':4s} {'8시갭':>7s} {'패턴상승':>6s} 예측  {'실제':>7s} 결과")
for i in rows:
    pp = predict(i, 20, "모양+크기", False, 1.0)[0]
    g = gap[i]; up = g > 0; agree = (pp > .5) == up; r = int(df.reg[i])
    grade = "강" if (agree and abs(g) >= .005) or (r == 1 and abs(g) >= .0075) else ("약" if not agree else "중")
    pred = "상" if up else "하"
    if np.isnan(df.y[i]) if isinstance(df.y[i], float) else False:
        print(f"{df.date[i].date()} {nm[r]} {g*100:+6.2f}% {pp*100:5.0f}%  {pred}({grade})   진행중"); continue
    real = df.post[i][-1]; ok = (real > 0) == up
    tot[grade][0] += ok; tot[grade][1] += 1
    print(f"{df.date[i].date()} {nm[r]} {g*100:+6.2f}% {pp*100:5.0f}%  {pred}({grade}) {real*100:+6.2f}% {'O' if ok else 'X'}")
h = sum(v[0] for v in tot.values()); n = sum(v[1] for v in tot.values())
print(f"\n전체 {h}/{n} = {h/n*100:.0f}%")
for k, (a, b) in tot.items():
    if b: print(f"  {k}: {a}/{b} = {a/b*100:.0f}%")
