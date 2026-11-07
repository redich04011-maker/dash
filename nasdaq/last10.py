"""최근 10거래일: 각 날짜 밤8시(KST)까지 정보만으로 그날 미국장 상/하 예측 → 실제와 비교
모델: 기존 지표 + 추세 (로지스틱), 매일 그 전날까지 데이터로만 다시 학습"""
import numpy as np, pandas as pd, yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from night import build, ET

nq = yf.download("NQ=F", period="4y", interval="1d", auto_adjust=True, progress=False)["Close"].squeeze()
nq.index = pd.to_datetime(nq.index).tz_localize(None)
ma20, ma60 = nq.rolling(20).mean(), nq.rolling(60).mean(); sl = ma20.pct_change(5)
reg = pd.Series(0, index=nq.index)
reg[(nq > ma20) & (sl > 0)] = 1
reg[(nq < ma20) & (sl < 0)] = -1
df = build(); df.index = pd.to_datetime(df.index)
df["reg"] = reg.shift(1).reindex(df.index); df = df.dropna()
df["gap_x_reg"] = df.nq_gap * df.reg
cols = ["nq_gap", "btc_gap", "dax_gap", "n225", "kospi", "dow", "p_nq", "p_sox", "p_vix", "p_tnx",
        "p_hy", "p_dxy", "p_nq5", "p_nq_ma20", "vol20", "reg", "gap_x_reg"]
now = pd.Timestamp.now(tz=ET)
done = [d for d in df.index if pd.Timestamp(f"{d.date()} 16:00", tz=ET) <= now]
live = [d for d in df.index if d not in done]
name = {1: "상승세", 0: "횡보세", -1: "하락세"}
print(f"{'미국장 날짜':10s} {'예측시각(KST)':14s} {'추세':4s} {'NQ갭':>7s} {'상승확률':>7s} {'예측':3s} {'실제':>7s} 결과")
hit = 0; rows = done[-10:] + live
for d in rows:
    i = df.index.get_loc(d)
    tr = df.iloc[:i]
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(tr[cols], tr.yA > 0)
    p = m.predict_proba(df.loc[[d], cols])[0, 1]
    r = df.loc[d]; pred = "상" if p > .5 else "하"
    conf = max(p, 1 - p)
    tag = "강" if conf >= .8 else ("중" if conf >= .7 else "약")
    if d in done:
        ok = (r.yA > 0) == (p > .5); hit += ok
        res = f"{r.yA*100:+6.2f}% {'O' if ok else 'X'}"
    else:
        res = "  진행중"
    print(f"{d.date()}  {d.date()} 20:00   {name[int(r.reg)]} {r.nq_gap*100:+6.2f}% {p*100:6.1f}%  {pred}({tag}) {res}")
print(f"\n최근 10일 적중 {hit}/10   (강=확신80%↑, 중=70~80%, 약=70%↓)")
