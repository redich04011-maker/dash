"""헤드라인 키워드가 '실제로' 도움이 되는지 판정 + 연관 키워드 자동 추출 (최근 1년)
- 하루 4개 검색어로 구글뉴스 RSS 수집, 발행시각이 '전일 장마감-8h ~ 당일 20:00 KST' 인 것만 (미래참조 없음)
- headlines.csv 캐시 (다음 실행부터는 새 날짜만 수집)
- walk-forward 비교: (기준) 갭 방향  vs  (모델) 갭+추세  vs  (모델) 갭+추세+키워드
- 키워드는 사람이 고르지 않고 단어/두단어 전체에서 '틀린 날' 연관도로 자동 순위
"""
import os, re, time, email.utils
import numpy as np, pandas as pd, requests
import xml.etree.ElementTree as X
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix
from pattern import df, gap, ET

Q = ["stock market", "nasdaq OR tech stocks", "federal reserve OR treasury yields OR inflation",
     "war OR tariffs OR geopolitical OR oil prices"]
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
d = df.copy(); d.index = pd.DatetimeIndex(d.date); d["gap"] = gap
days = list(d.index[-262:])
CACHE = "headlines.csv"
cache = pd.read_csv(CACHE) if os.path.exists(CACHE) else pd.DataFrame({"date": [], "q": [], "title": []})
cache["date"] = pd.to_datetime(cache["date"])
have = set(cache.date.dt.date) if len(cache) else set()
today = d.index[-1].date()


def fetch(day):
    i = d.index.get_loc(day); prev = d.index[i - 1]
    lo = pd.Timestamp(f"{prev.date()} 16:00", tz=ET).tz_convert("UTC") - pd.Timedelta(hours=8)
    hi = pd.Timestamp(f"{day.date()} 11:00", tz="UTC")
    out = []
    for q in Q:
        qq = f"({q}) after:{(prev - pd.Timedelta(days=1)).date()} before:{(day + pd.Timedelta(days=1)).date()}"
        for _ in range(3):
            try:
                r = requests.get("https://news.google.com/rss/search", params={"q": qq, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                                 headers=UA, timeout=20)
                if r.status_code == 200: break
            except Exception: pass
            time.sleep(3)
        else: continue
        for it in X.fromstring(r.content).iter("item"):
            try: ts = pd.Timestamp(email.utils.parsedate_to_datetime(it.findtext("pubDate"))).tz_convert("UTC")
            except Exception: continue
            if lo <= ts <= hi: out.append((day, q, re.sub(r"\s+-\s+[^-]+$", "", it.findtext("title") or "")))
        time.sleep(0.4)
    return out


new = []


def save():
    global cache, new
    if not new: return
    add = pd.DataFrame(new, columns=["date", "q", "title"]); add["date"] = pd.to_datetime(add["date"])
    cache = pd.concat([cache[cache.date.dt.date != today], add]).drop_duplicates(["date", "title"])
    cache.to_csv(CACHE, index=False); new = []


for n, day in enumerate(days):
    if day.date() in have and day.date() != today: continue
    new += fetch(day)
    if n % 20 == 0: save(); print(f"  수집 {n+1}/{len(days)}일", flush=True)
save()
txt = cache.groupby(cache.date.dt.normalize()).title.apply(lambda s: " . ".join(s)).reindex(days).fillna("")
cnt = cache.groupby(cache.date.dt.normalize()).size().reindex(days).fillna(0)
print(f"헤드라인 {len(cache)}개, {len(days)}일, 하루 평균 {cnt.mean():.0f}개 (최소 {cnt.min():.0f})\n")

D = d.loc[days].copy(); D["txt"] = txt.values
done = D.y.notna().values
y = np.where(done, D.y, 0).astype(bool)
g = D.gap.values; r = D.reg.values.astype(float)
miss = ((g > 0) != y) & done
num = np.c_[g / .01, np.abs(g) / .01, r, g / .01 * r]

vec = CountVectorizer(ngram_range=(1, 2), min_df=6, max_df=.5, binary=True, stop_words="english",
                      token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-']+\b")
W = vec.fit_transform(D.txt)          # 어휘만 전체에서 정함(라벨 미사용) — 가중치는 walk-forward로만 학습
voc = np.array(vec.get_feature_names_out())
print(f"키워드 후보 {len(voc)}개\n")


def wf(Xf, C, start=120, step=10):
    P = np.full(len(y), np.nan)
    for s in range(start, len(y), step):
        tr = np.where(done[:s])[0]
        m = LogisticRegression(C=C, max_iter=5000).fit(Xf[tr], y[tr])
        P[s:s + step] = m.predict_proba(Xf[s:s + step])[:, 1]
    return P


k = done & (np.arange(len(y)) >= 120)
print(f"[판정] walk-forward 검증 {k.sum()}일 (최근 약 {k.sum()/21:.0f}개월)")
print(f"  갭 방향만                {((g > 0) == y)[k].mean()*100:5.1f}%")
P0 = wf(num, 1.0)
print(f"  갭+추세 모델              {((P0 > .5) == y)[k].mean()*100:5.1f}%")
best = None
for C in (0.02, 0.05, 0.1):
    Xf = hstack([csr_matrix(num), W * 0.5]).tocsr()
    P1 = wf(Xf, C)
    a = ((P1 > .5) == y)[k].mean()
    print(f"  갭+추세+키워드 (C={C})     {a*100:5.1f}%")
    if best is None or a > best[0]: best = (a, C, P1)
a, C, P1 = best
c0 = np.abs(P0 - .5) + .5; c1 = np.abs(P1 - .5) + .5
for th in (.7, .8):
    m0, m1 = k & (c0 >= th), k & (c1 >= th)
    print(f"  확신≥{int(th*100)}%: 갭+추세 {((P0>.5)==y)[m0].mean()*100:5.1f}%({m0.sum()}일) | +키워드 {((P1>.5)==y)[m1].mean()*100:5.1f}%({m1.sum()}일)")
flip = k & ((P1 > .5) != (g > 0))
print(f"  키워드가 갭 방향을 뒤집은 날 {flip.sum()}일: 뒤집기 적중 {((P1>.5)==y)[flip].mean()*100 if flip.sum() else 0:.1f}%")

# 연관 키워드 (설명용, 전체 기간) : 틀린 날 비율 리프트
base = miss[done].mean()
dfreq = np.asarray(W[done].sum(0)).ravel()
mcount = np.asarray(W[done][miss[done]].sum(0)).ravel()
rate = (mcount + base * 4) / (dfreq + 4)          # 스무딩
ok = dfreq >= 8
order = np.argsort(-(rate - base) * ok)
print(f"\n[틀린 날과 연관된 키워드 상위] 기준 틀린비율 {base*100:.1f}%")
for j in order[:15]:
    if ok[j]: print(f"  {voc[j]:24s} 틀림 {rate[j]*100:5.1f}% ({int(dfreq[j])}일)")
print("\n[맞은 날과 연관된 키워드(갭 신뢰↑) 상위]")
for j in np.argsort((rate - base) + (~ok) * 9)[:15]:
    print(f"  {voc[j]:24s} 틀림 {rate[j]*100:5.1f}% ({int(dfreq[j])}일)")

m = LogisticRegression(C=C, max_iter=5000).fit(hstack([csr_matrix(num), W * .5]).tocsr()[done], y[done])
coef = m.coef_[0][4:]
print("\n[모델이 학습한 방향 키워드] +는 상승 쪽, -는 하락 쪽")
print("  상승: " + ", ".join(voc[np.argsort(-coef)[:12]]))
print("  하락: " + ", ".join(voc[np.argsort(coef)[:12]]))
i = len(y) - 1
print(f"\n오늘 {D.index[i].date()}: 갭 {g[i]*100:+.2f}%, 갭+추세 상승확률 {P0[i]*100:.0f}% → +키워드 {P1[i]*100:.0f}%  (헤드라인 {int(cnt.iloc[i])}개)")
hit = [voc[j] for j in W[i].indices if abs(coef[j]) > np.quantile(np.abs(coef), .9)]
print("  오늘 영향 키워드: " + ", ".join(hit[:15]))
