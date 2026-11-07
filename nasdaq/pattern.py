"""패턴(모양) 확률: 전일 16:00 ET → 당일 20:00 KST 나스닥선물 흐름을 과거 비슷한 날과 비교
- 모양: 15개 지점으로 맞춘 누적등락 곡선
- 비슷함: (1) 모양+크기(유클리드)  (2) 모양만(정규화 후 거리)
- 판정: 가장 비슷한 K일 중 상승 마감 비율
- 검증: 무작위 샘플 날짜를 여러 번 뽑아, 각 날짜보다 '과거' 날만 후보로 사용 (미래참조 없음)
- 차트: 오늘 모양(굵은선) + 비슷한 과거 20일과 그날 8시 이후 흐름
"""
import numpy as np, pandas as pd, yfinance as yf
from night import hourly, asof, ET

N_PRE, N_POST = 15, 12
nq = hourly("NQ=F")
d = yf.download("NQ=F", period="4y", interval="1d", auto_adjust=True, progress=False)["Close"].squeeze()
d.index = pd.to_datetime(d.index).tz_localize(None)
ma20 = d.rolling(20).mean(); sl = ma20.pct_change(5)
reg = pd.Series(0, index=d.index); reg[(d > ma20) & (sl > 0)] = 1; reg[(d < ma20) & (sl < 0)] = -1
reg = reg.shift(1)
vix = yf.download("^VIX", period="4y", interval="1d", auto_adjust=True, progress=False)["Close"].squeeze()
vix.index = pd.to_datetime(vix.index).tz_localize(None)
ctx_daily = pd.DataFrame({"vol20": d.pct_change().rolling(20).std(), "ma20gap": d / ma20 - 1,
                          "vix": vix.reindex(d.index).ffill()}).shift(1)
ym = d.index.to_period("M"); first = pd.Series(d.index, index=d.index).groupby(ym).transform("min")
pos = {x: k for k, x in enumerate(d.index)}
mstart = pd.Series([pos[x] - pos[f] <= 2 for x, f in zip(d.index, first)], index=d.index).astype(int)

us = [x.date() for x in d.index if x >= nq.index.min().tz_convert(None)]
now = pd.Timestamp.now(tz="UTC")
rows = []
for i in range(1, len(us)):
    d0, d1 = us[i - 1], us[i]
    c0 = pd.Timestamp(f"{d0} 16:00", tz=ET).tz_convert("UTC")
    c1 = pd.Timestamp(f"{d1} 16:00", tz=ET).tz_convert("UTC")
    T = pd.Timestamp(f"{d1} 11:00", tz="UTC")
    if T > now: continue
    base = asof(nq, c0)
    pre = [asof(nq, c0 + (T - c0) * k / (N_PRE - 1)) / base - 1 for k in range(N_PRE)]
    done = c1 <= now
    post = [asof(nq, T + (c1 - T) * k / N_POST) / base - 1 for k in range(1, N_POST + 1)] if done else [np.nan] * N_POST
    rows.append(dict(date=pd.Timestamp(d1), pre=np.array(pre), post=np.array(post),
                     y=(post[-1] > 0) if done else np.nan, reg=reg.get(pd.Timestamp(d1), np.nan),
                     dow=pd.Timestamp(d1).dayofweek, mstart=mstart.get(pd.Timestamp(d1), 0),
                     **ctx_daily.loc[pd.Timestamp(d1)].to_dict()))
df = pd.DataFrame(rows).dropna(subset=["reg"]).reset_index(drop=True)
df = df[df.pre.apply(lambda a: np.isfinite(a).all())].reset_index(drop=True)
P = np.stack(df.pre.values)
gap = P[:, -1]
shape = (P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-9)
full = df[df.y.notna()].index.values
CN = np.stack([(df[c] - df[c].mean()) / df[c].std() for c in ("vol20", "ma20gap", "vix")], 1)  # 연속형 상황
CC = df[["reg", "dow", "mstart"]].values                                                   # 범주형 상황
print(f"{df.date.iloc[0].date()} ~ {df.date.iloc[-1].date()}  {len(df)}일 (결과 확정 {len(full)}일)\n")


def predict(i, k, mode, same_reg=False, w=0.0):
    cand = np.array([j for j in full if df.date[j] < df.date[i]])
    if same_reg: cand = cand[df.reg.values[cand] == df.reg[i]]
    if len(cand) < k: return np.nan, np.nan
    X = P if mode == "모양+크기" else shape
    dist = np.sqrt(((X[cand] - X[i]) ** 2).sum(1))
    if w > 0:  # 상황 유사도: 연속형 거리 + 범주 불일치 개수
        cd = np.sqrt(((CN[cand] - CN[i]) ** 2).sum(1)) + (CC[cand] != CC[i]).sum(1)
        dist = dist / np.median(dist) + w * cd / np.median(cd)
    nb = cand[np.argsort(dist)[:k]]
    return df.y.values[nb].astype(float).mean(), nb


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    pool = full[full >= 120]  # 앞 120일은 후보 풀로만 사용
    print(f"[검증] 무작위 150일 × 5회 샘플링 (후보: 그날 이전 날만)\n")
    print(f"  {'방식':22s} {'K':>3s}  {'전체적중':>8s}  {'일치율≥70%':>14s}  {'일치율≥80%':>14s}")
    for mode in ("모양+크기", "모양만"):
        for w in (0.0, 0.5, 1.0, 2.0):
            for k in (20, 40):
                acc, a70, n70, a80, n80 = [], [], [], [], []
                for s_ in range(5):
                    test = np.random.default_rng(s_).choice(pool, 150, replace=False)
                    pr = np.array([predict(i, k, mode, False, w)[0] for i in test])
                    yy = df.y.values[test].astype(bool); ok = ~np.isnan(pr)
                    hit = (pr > .5) == yy; agree = np.maximum(pr, 1 - pr)
                    acc.append(hit[ok].mean())
                    m = ok & (agree >= .7); a70.append(hit[m].mean() if m.sum() else np.nan); n70.append(m.sum())
                    m = ok & (agree >= .8); a80.append(hit[m].mean() if m.sum() else np.nan); n80.append(m.sum())
                nm = f"{mode} 상황가중{w}"
                print(f"  {nm:22s} {k:3d}  {np.mean(acc)*100:5.1f}±{np.std(acc)*100:3.1f}  "
                      f"{np.nanmean(a70)*100:5.1f}%({np.mean(n70):4.0f}일)  {np.nanmean(a80)*100:5.1f}%({np.mean(n80):4.0f}일)")
    yy = df.y.values[pool].astype(bool)
    print(f"\n  [비교] 8시 갭 방향만: {((gap[pool] > 0) == yy).mean()*100:.1f}%")

    # 패턴 + 갭 결합
    print("\n[결합] 갭 방향과 패턴(모양+크기+상황, K=20) 방향이 일치하는 날")
    pr = np.array([predict(i, 20, "모양+크기", False, 1.0)[0] for i in pool])
    ok = ~np.isnan(pr); same_dir = ok & ((pr > .5) == (gap[pool] > 0))
    print(f"  일치 {same_dir.sum()}일: 적중 {((gap[pool]>0)==yy)[same_dir].mean()*100:.1f}% | "
          f"불일치 {(ok & ~same_dir).sum()}일: 갭쪽 적중 {((gap[pool]>0)==yy)[ok & ~same_dir].mean()*100:.1f}%")
    for lo in (0.0025, 0.005):
        m = same_dir & (np.abs(gap[pool]) >= lo)
        print(f"  일치 + |갭|≥{lo*100:.2f}%: {((gap[pool]>0)==yy)[m].mean()*100:.1f}% ({m.sum()}일)")

    # 오늘 차트
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "NanumGothic"; plt.rcParams["axes.unicode_minus"] = False
    i = len(df) - 1
    p_up, nb = predict(i, 20, "모양+크기", False, 1.0)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=130)
    xp = np.arange(N_PRE); xa = np.arange(N_PRE - 1, N_PRE + N_POST)
    for j in nb:
        up = df.y[j]; c = "#d33" if up else "#26c"
        ax.plot(xp, P[j] * 100, color=c, alpha=.35, lw=1)
        ax.plot(xa, np.r_[P[j][-1], df.post[j]] * 100, color=c, alpha=.35, lw=1, ls="--")
    ax.plot(xp, P[i] * 100, color="black", lw=4, label=f"오늘 {df.date[i].date()} (8시까지)")
    ax.axvline(N_PRE - 1, color="gray", ls=":"); ax.axhline(0, color="gray", lw=.8)
    ax.text(N_PRE - 1, ax.get_ylim()[1], " 밤8시", va="top", color="gray")
    ax.set_xticks([0, N_PRE - 1, N_PRE + N_POST - 1]); ax.set_xticklabels(["전일종가", "20:00 KST", "당일종가"])
    ax.set_ylabel("전일종가 대비 %")
    nm = {1: "상승세", 0: "횡보세", -1: "하락세"}[int(df.reg[i])]
    ax.set_title(f"모양+상황 비슷한 과거 20일 (오늘 {nm}) → 상승 마감 {p_up*100:.0f}%  (빨강=상승마감, 파랑=하락마감, 점선=8시 이후)")
    ax.legend(loc="lower left"); fig.tight_layout(); fig.savefig("pattern_today.png")
    print(f"\n오늘 {df.date[i].date()} ({nm}): 갭 {gap[i]*100:+.2f}%, 비슷한 20일 중 상승 마감 {p_up*100:.0f}%")
