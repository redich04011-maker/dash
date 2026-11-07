"""개선 등급(8시 정보만) — 2년 전체 + 최근 한 달
방향 = 8시 NQ갭 부호
강: (|갭|≥0.75%) OR (상승갭 AND 20일선 이격 상위1/3) OR (갭·패턴일치 AND |갭|≥0.5%)
약: (|갭|<0.25%) OR (VIX≥25) OR (월초3일 AND 하락갭) OR (월요일) OR (갭·패턴 엇갈림)   ※강이 우선
중: 그 외
A = 전일종가→당일종가 방향 / B = 밤8시→당일종가 방향(8시 매수 시 실제 손익)"""
import numpy as np, pandas as pd
from pattern import df, gap, predict

q67 = df.ma20gap.quantile(.67)
rows = []
for i in range(120, len(df)):
    if pd.isna(df.y[i]): continue
    g = gap[i]; up = g > 0
    pp = predict(i, 20, "모양+크기", False, 1.0)[0]; agree = (pp > .5) == up
    strong = abs(g) >= .0075 or (up and df.ma20gap[i] >= q67) or (agree and abs(g) >= .005)
    weak = abs(g) < .0025 or df.vix[i] >= 25 or (df.mstart[i] and not up) or df.dow[i] == 0 or not agree
    grade = "강" if strong else ("약" if weak else "중")
    a = df.post[i][-1]; bret = (1 + a) / (1 + g) - 1
    rows.append(dict(date=df.date[i], g=g, grade=grade, A=(a > 0) == up, B=(bret > 0) == up,
                     bret=bret * (1 if up else -1)))
r = pd.DataFrame(rows)
def show(x, title):
    print(f"[{title}] {len(x)}일")
    print(f"  {'등급':4s} {'일수':>4s} {'A 종가방향':>10s} {'B 8시매수후':>11s} {'B 평균손익':>9s}")
    for k in ("강", "중", "약"):
        s = x[x.grade == k]
        if len(s): print(f"  {k:4s} {len(s):4d} {s.A.mean()*100:9.1f}% {s.B.mean()*100:10.1f}% {s.bret.mean()*100:+8.2f}%")
    print(f"  전체 {len(x):4d} {x.A.mean()*100:9.1f}% {x.B.mean()*100:10.1f}% {x.bret.mean()*100:+8.2f}%\n")
show(r, "2년 검증")
show(r.tail(21), "최근 한 달")
print("최근 한 달 상세")
for _, x in r.tail(21).iterrows():
    print(f"  {x.date.date()} 갭 {x.g*100:+.2f}% {'상' if x.g>0 else '하'}({x.grade})  A {'O' if x.A else 'X'}  B {'O' if x.B else 'X'} ({x.bret*100:+.2f}%)")
