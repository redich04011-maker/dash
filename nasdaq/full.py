"""밤8시 상/하 — 기존 지표 + 추세 + 날짜효과 + 추가지표 (FOMC 제외, 최근 2년)"""
import numpy as np, pandas as pd, yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from night import build

dl = yf.download(["NQ=F", "NVDA", "^VIX", "^VIX3M"], period="4y", interval="1d",
                 auto_adjust=True, progress=False)["Close"]
dl.index = pd.to_datetime(dl.index).tz_localize(None)
nq = dl["NQ=F"]
ma20, ma60 = nq.rolling(20).mean(), nq.rolling(60).mean()
sl = ma20.pct_change(5)
reg = pd.Series(0, index=nq.index)
reg[(nq > ma20) & (sl > 0)] = 1
reg[(nq < ma20) & (sl < 0)] = -1
extra = pd.DataFrame({
    "reg": reg, "p_nvda": dl["NVDA"].pct_change(),
    "vix_term": dl["^VIX"] / dl["^VIX3M"],              # >1 이면 단기공포 역전
}).shift(1)                                            # 전일까지 정보

df = build()
df.index = pd.to_datetime(df.index)
df = df.join(extra, how="left").dropna()

# ---- 날짜 효과 (미리 알 수 있는 정보) ----
days = nq.dropna().index
pos = {d: i for i, d in enumerate(days)}
ym = days.to_period("M")
first = pd.Series(days, index=days).groupby(ym).transform("min")
last = pd.Series(days, index=days).groupby(ym).transform("max")
rank_s = pd.Series([pos[d] - pos[f] for d, f in zip(days, first)], index=days)
rank_e = pd.Series([pos[l] - pos[d] for d, l in zip(days, last)], index=days)
cal = pd.DataFrame(index=days)
cal["월초3일"] = (rank_s <= 2).astype(int)
cal["월말2일"] = (rank_e <= 1).astype(int)
cal["옵션만기"] = ((days.weekday == 4) & (days.day >= 15) & (days.day <= 21)).astype(int)
cal["고용지표일"] = ((days.weekday == 4) & (days.day <= 7)).astype(int)
nxt = pd.Series(days[1:].append(pd.DatetimeIndex([days[-1] + pd.Timedelta(days=1)])), index=days)
cal["휴일전날"] = ((nxt - days.to_series()).dt.days > 3).astype(int) | \
                 (((nxt - days.to_series()).dt.days > 1) & (days.weekday != 4)).astype(int)
cal["연말연초"] = (((days.month == 12) & (days.day >= 20)) | ((days.month == 1) & (days.day <= 5))).astype(int)
for i, n in enumerate(["월", "화", "수", "목", "금"]):
    cal[n + "요일"] = (days.weekday == i).astype(int)
df = df.join(cal, how="left").fillna(0)
df["gap_x_reg"] = df.nq_gap * df.reg

# ---- 보조지표 (전일 일봉까지) ----
o = yf.download("NQ=F", period="4y", interval="1d", auto_adjust=True, progress=False)
o.columns = o.columns.get_level_values(0); o.index = pd.to_datetime(o.index).tz_localize(None)
C, Hh, Ll = o["Close"], o["High"], o["Low"]
def rsi(x, n):
    d = x.diff(); u = d.clip(lower=0).ewm(alpha=1/n).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/n).mean()
    return 100 - 100 / (1 + u / dn)
macd = C.ewm(span=12).mean() - C.ewm(span=26).mean()
bb_m, bb_s = C.rolling(20).mean(), C.rolling(20).std()
up = (C.diff() > 0).astype(int)
streak = up.groupby((up != up.shift()).cumsum()).cumcount() + 1
tr = pd.concat([Hh - Ll, (Hh - C.shift()).abs(), (Ll - C.shift()).abs()], axis=1).max(axis=1)
T = pd.DataFrame({
    "t_ret1": C.pct_change(), "t_ret2": C.pct_change(2), "t_ret3": C.pct_change(3),
    "t_rsi14": rsi(C, 14), "t_rsi2": rsi(C, 2),
    "t_macdh": (macd - macd.ewm(span=9).mean()) / C,
    "t_bbpct": (C - bb_m) / (2 * bb_s),
    "t_stoch": (C - Ll.rolling(14).min()) / (Hh.rolling(14).max() - Ll.rolling(14).min()),
    "t_streak": np.where(up == 1, streak, -streak),
    "t_closeloc": (C - Ll) / (Hh - Ll),                      # 전일 종가의 고저 내 위치
    "t_range_atr": (Hh - Ll) / tr.rolling(14).mean(),
    "t_hi20": C / Hh.rolling(20).max() - 1, "t_lo20": C / Ll.rolling(20).min() - 1,
}).shift(1)
df = df.join(T, how="left").dropna()
tcols = list(T.columns)

y = (df.yA > 0).values
print("[보조지표] 오늘 종가방향과 상관 / 밤8시이후(B)와 상관 / 상위·하위 20% 상승비율")
for c in tcols:
    q1, q4 = df[c].quantile(.2), df[c].quantile(.8)
    print(f"  {c:12s} A {np.corrcoef(df[c], df.yA)[0,1]:+.3f}  B {np.corrcoef(df[c], df.yB)[0,1]:+.3f}  "
          f"하위20% 상승 {(df.yA[df[c]<=q1]>0).mean()*100:4.1f}%  상위20% 상승 {(df.yA[df[c]>=q4]>0).mean()*100:4.1f}%")
print()
print(f"{df.index[0].date()} ~ {df.index[-1].date()}  {len(df)}거래일, 전체 상승일 {y.mean()*100:.1f}%\n")

print("[날짜효과] 해당일 상승비율 / 갭방향 적중률")
g = df.nq_gap.values
for c in ["월초3일", "월말2일", "옵션만기", "고용지표일", "휴일전날", "연말연초", "월요일", "화요일", "수요일", "목요일", "금요일"]:
    m = df[c].values == 1
    if m.sum(): print(f"  {c:6s} {m.sum():3d}일  상승 {y[m].mean()*100:5.1f}%  갭적중 {((g[m]>0)==y[m]).mean()*100:5.1f}%")

print("\n[추가지표 상관] 전일NVDA %.3f | VIX기간구조 %.3f" %
      (np.corrcoef(df.p_nvda, df.yA)[0, 1], np.corrcoef(df.vix_term, df.yA)[0, 1]))

def wf(cols):
    X = df[cols].values.astype(float); P = np.full(len(y), np.nan)
    for s in range(200, len(y), 20):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(X[:s], y[:s])
        P[s:s+20] = m.predict_proba(X[s:s+20])[:, 1]
    return P

old = ["nq_gap", "btc_gap", "dax_gap", "n225", "kospi", "dow", "p_nq", "p_sox", "p_vix", "p_tnx",
       "p_hy", "p_dxy", "p_nq5", "p_nq_ma20", "vol20"]
sets = {
    "기존": old,
    "기존+추세": old + ["reg", "gap_x_reg"],
    "전부": old + ["reg", "gap_x_reg", "p_nvda", "vix_term", "월초3일", "월말2일", "옵션만기",
                  "고용지표일", "휴일전날", "연말연초", "월요일", "화요일", "수요일", "목요일"],
}
sets["전부+보조지표"] = sets["전부"] + tcols
sets["갭+보조지표만"] = ["nq_gap", "reg", "gap_x_reg"] + tcols
print("\n[모델 비교] 확신구간별 적중률 (walk-forward, 검증 %d일)" % (len(y) - 200))
bands = [(0.9, 1.01), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.5, 0.6)]
print("  모델        전체    " + "  ".join(f"{int(a*100)}~{min(int(b*100),100)}%" for a, b in bands))
for name, cols in sets.items():
    P = wf(cols); k = ~np.isnan(P); c = np.abs(P - .5) + .5; h = (P > .5) == y
    row = f"  {name:8s} {h[k].mean()*100:5.1f}%  "
    for a, b in bands:
        m = k & (c >= a) & (c < b)
        row += f"{h[m].mean()*100:5.1f}%({m.sum():3d}) " if m.sum() else "     -      "
    print(row)
