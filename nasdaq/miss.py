"""틀린 날 연구: 밤8시 갭 방향 ≠ 당일 종가 방향인 날의 특징 + 뒤집힌 시각"""
import numpy as np, pandas as pd
from pattern import df, gap, predict, nq, asof, ET
from night import build

b = build(); b.index = pd.to_datetime(b.index)
d = df.set_index("date").join(b[["dax_gap", "n225", "kospi", "btc_gap", "p_nq", "p_nq5"]], how="left")
d["gap"] = gap; d = d[d.y.notna()].copy()
d["y"] = d.y.astype(bool); d["miss"] = (d.gap > 0) != d.y
d["absgap"] = d.gap.abs()
print(f"{len(d)}일 중 틀린 날 {d.miss.sum()}일 ({d.miss.mean()*100:.1f}%)\n")

def tab(title, key):
    t = d.groupby(key, observed=True).miss.agg(["mean", "size"])
    print(f"[{title}]  틀린 비율")
    for k, (m, n) in t.iterrows(): print(f"  {str(k):18s} {m*100:5.1f}%  ({n}일)")
    print()

tab("갭 크기", pd.cut(d.absgap * 100, [0, .1, .25, .5, .75, 1, 99]))
tab("추세", d.reg.map({1: "상승세", 0: "횡보세", -1: "하락세"}))
tab("요일", d.index.dayofweek.map(dict(enumerate("월화수목금"))))
tab("월초 3일", d.mstart.map({1: "월초", 0: "그외"}))
tab("변동성(20일)", pd.qcut(d.vol20, 3, labels=["낮음", "중간", "높음"]))
tab("VIX", pd.cut(d.vix, [0, 15, 20, 25, 99], labels=["<15", "15~20", "20~25", "25+"]))
tab("20일선 이격", pd.qcut(d.ma20gap, 3, labels=["아래", "근처", "위"]))
tab("DAX 방향", np.where(np.sign(d.dax_gap) == np.sign(d.gap), "갭과 같음", "갭과 반대"))
tab("아시아(닛케이) 방향", np.where(np.sign(d.n225) == np.sign(d.gap), "갭과 같음", "갭과 반대"))
tab("BTC 방향", np.where(np.sign(d.btc_gap) == np.sign(d.gap), "갭과 같음", "갭과 반대"))
tab("전일 나스닥 방향", np.where(np.sign(d.p_nq) == np.sign(d.gap), "갭과 같음(추격)", "갭과 반대(반등/반락)"))
tab("갭 방향", np.where(d.gap > 0, "상승갭", "하락갭"))

# 갭 작은 날 제외 후 조합
big = d[d.absgap >= .0025]
print(f"[갭≥0.25% 날만 {len(big)}일] 조합별 틀린 비율")
for nm, m in {
    "상승갭+하락세": (big.gap > 0) & (big.reg == -1),
    "하락갭+상승세": (big.gap < 0) & (big.reg == 1),
    "상승갭+20일선 위(과열)": (big.gap > 0) & (big.ma20gap > big.ma20gap.quantile(.67)),
    "하락갭+20일선 아래(과매도)": (big.gap < 0) & (big.ma20gap < big.ma20gap.quantile(.33)),
    "갭이 전일과 반대(반등시도)": np.sign(big.p_nq) != np.sign(big.gap),
    "DAX 반대": np.sign(big.dax_gap) != np.sign(big.gap),
    "VIX 25+": big.vix >= 25,
}.items():
    print(f"  {nm:24s} {big.miss[m].mean()*100:5.1f}% ({m.sum()}일)   그외 {big.miss[~m].mean()*100:5.1f}%")

# 뒤집힌 시각
print("\n[틀린 날: 전일종가선을 마지막으로 넘은 시각(KST)] = 방향이 확정적으로 바뀐 때")
bins = {"20~22:30 유럽장": 0, "22:30~23:30 개장전": 0, "개장 첫1시간": 0, "장중": 0, "마감 1시간": 0}
for dt, r in d[d.miss].iterrows():
    c0 = pd.Timestamp(f"{(dt - pd.tseries.offsets.BDay(1)).date()} 16:00", tz=ET).tz_convert("UTC")
    base = asof(nq, c0)
    T = pd.Timestamp(f"{dt.date()} 11:00", tz="UTC"); c1 = pd.Timestamp(f"{dt.date()} 16:00", tz=ET).tz_convert("UTC")
    s = nq[(nq.index + pd.Timedelta(hours=1) > T) & (nq.index + pd.Timedelta(hours=1) <= c1)] / base - 1
    side = np.sign(s.iloc[-1]); cross = s[np.sign(s) != side]
    t = (cross.index[-1] + pd.Timedelta(hours=1)) if len(cross) else T
    et = t.tz_convert(ET); h = et.hour + et.minute / 60
    k = ("20~22:30 유럽장" if h < 8.5 else "22:30~23:30 개장전" if h < 9.5 else
         "개장 첫1시간" if h < 10.5 else "마감 1시간" if h >= 15 else "장중")
    bins[k] += 1
for k, v in bins.items(): print(f"  {k:20s} {v:3d}일 ({v/d.miss.sum()*100:.0f}%)")

print("\n[틀린 날 크기] 실제 종가 등락 |%|: 평균 "
      f"{d[d.miss].post.apply(lambda a: abs(a[-1])).mean()*100:.2f}% vs 맞은 날 {d[~d.miss].post.apply(lambda a: abs(a[-1])).mean()*100:.2f}%")

# 최근 틀린 날 차트
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "NanumGothic"; plt.rcParams["axes.unicode_minus"] = False
recent = d[d.miss].index[-6:]
fig, axs = plt.subplots(2, 3, figsize=(14, 7), dpi=120)
for ax, dt in zip(axs.flat, recent):
    c0 = pd.Timestamp(f"{(dt - pd.tseries.offsets.BDay(1)).date()} 16:00", tz=ET).tz_convert("UTC")
    c1 = pd.Timestamp(f"{dt.date()} 16:00", tz=ET).tz_convert("UTC")
    s = nq[(nq.index + pd.Timedelta(hours=1) > c0) & (nq.index + pd.Timedelta(hours=1) <= c1)]
    s = (s / asof(nq, c0) - 1) * 100; x = (s.index + pd.Timedelta(hours=1)).tz_convert("Asia/Seoul")
    ax.plot(x, s.values, color="black", lw=2); ax.axhline(0, color="gray", lw=.8)
    ax.axvline(pd.Timestamp(f"{dt.date()} 20:00", tz="Asia/Seoul"), color="orange", ls="--", label="밤8시")
    ax.axvline(pd.Timestamp(f"{dt.date()} 09:30", tz=ET).tz_convert("Asia/Seoul"), color="green", ls=":", label="미국개장")
    r = d.loc[dt]
    ax.set_title(f"{dt.date()} {['상승세','횡보세','하락세'][[1,0,-1].index(int(r.reg))]} 갭{r.gap*100:+.2f}% → 종가{r.post[-1]*100:+.2f}%", fontsize=10)
    ax.tick_params(axis="x", labelsize=7); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H시", tz="Asia/Seoul"))
axs.flat[0].legend(fontsize=8); fig.suptitle("최근 틀린 날: 전일종가 대비 나스닥선물 흐름 (KST)"); fig.tight_layout()
fig.savefig("miss_recent.png")
