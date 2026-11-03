"""틀린 날 거꾸로 이슈 찾기
- 틀린 날 / 맞은 날 표본 → 그날 '밤8시(KST) 이전' 뉴스 헤드라인 수집(구글뉴스 RSS, 발행시각으로 컷)
- 헤드라인을 이슈 범주로 분류 → 틀린 날에 유독 많은 이슈(위험 이슈) 찾기
- 위험 이슈가 있는 날 '보류/약' 처리 시 적중률 변화 확인
"""
import re, time, email.utils
import numpy as np, pandas as pd, requests
import xml.etree.ElementTree as ET_
from pattern import df, gap, ET

CAT = {
    "전쟁·지정학": r"\b(war|missile|strike[sd]?|attack|invasion|troops|military|iran|israel|gaza|hamas|hezbollah|ukraine|russia|taiwan|north korea|ceasefire|drone)\b",
    "관세·무역": r"\b(tariffs?|trade war|trade deal|duties|import tax|export (ban|control)s?|sanctions?)\b",
    "연준·금리": r"\b(fed|fomc|powell|rate cut|rate hike|interest rates?|central bank|dovish|hawkish)\b",
    "물가": r"\b(cpi|inflation|ppi|consumer prices|pce)\b",
    "고용·경기": r"\b(jobs report|payrolls?|unemployment|jobless|recession|gdp|layoffs?)\b",
    "실적": r"\b(earnings|quarterly results|revenue|guidance|profit)\b",
    "AI·반도체": r"\b(nvidia|chips?|semiconductors?|ai|artificial intelligence|tsmc|openai)\b",
    "정치·정부": r"\b(trump|white house|congress|shutdown|election|senate|biden|executive order|debt ceiling)\b",
    "은행·신용": r"\b(banks?|credit|default|bankruptcy|liquidity|bond market)\b",
    "유가·원자재": r"\b(oil|crude|opec|gold|commodit(y|ies))\b",
    "중국": r"\b(china|chinese|beijing|xi jinping)\b",
    "엔화·일본": r"\b(yen|boj|bank of japan|japan)\b",
    "국채금리": r"\b(treasury yields?|10-year|bond yields?)\b",
    "급락·공포": r"\b(plunge[sd]?|crash|sell-?off|tumble[sd]?|rout|panic|fear|volatility|vix)\b",
}

d = df.copy(); d.index = pd.DatetimeIndex(d.date); d["gap"] = gap
d = d[d.y.notna()].copy(); d["y"] = d.y.astype(bool); d["miss"] = (d.gap > 0) != d.y
rng = np.random.default_rng(7)
miss_days = d[d.miss].index; hit_days = d[~d.miss].index
pick = sorted(list(rng.choice(miss_days, min(110, len(miss_days)), replace=False)) +
              list(rng.choice(hit_days, min(110, len(hit_days)), replace=False)))
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}


def heads(day):
    day = pd.Timestamp(day)
    prev = d.index[d.index.get_loc(day) - 1] if d.index.get_loc(day) > 0 else day - pd.Timedelta(days=1)
    c0 = pd.Timestamp(f"{prev.date()} 16:00", tz=ET).tz_convert("UTC")
    T = pd.Timestamp(f"{day.date()} 11:00", tz="UTC")
    q = f"(stock market OR nasdaq OR wall street OR stocks) after:{(prev - pd.Timedelta(days=1)).date()} before:{(day + pd.Timedelta(days=1)).date()}"
    for _ in range(3):
        try:
            r = requests.get("https://news.google.com/rss/search", params={"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                             headers=UA, timeout=20)
            if r.status_code == 200: break
        except Exception: pass
        time.sleep(3)
    else:
        return None
    out = []
    for it in ET_.fromstring(r.content).iter("item"):
        try: ts = pd.Timestamp(email.utils.parsedate_to_datetime(it.findtext("pubDate"))).tz_convert("UTC")
        except Exception: continue
        if c0 - pd.Timedelta(hours=8) <= ts <= T:     # 전일 장마감 전후 ~ 당일 밤8시 KST
            out.append(it.findtext("title") or "")
    return out


rows = []
for k, day in enumerate(pick):
    h = heads(day)
    if h is None: continue
    text = " | ".join(h).lower()
    rec = {"date": pd.Timestamp(day).date(), "miss": bool(d.miss[day]), "n_heads": len(h), "heads": " || ".join(h[:15])}
    for c, pat in CAT.items():
        rec[c] = len(re.findall(pat, text))
    rows.append(rec); time.sleep(0.7)
r = pd.DataFrame(rows)
r.to_csv("news_sample.csv", index=False, encoding="utf-8-sig")
print(f"수집 {len(r)}일 (틀린 날 {r.miss.sum()}, 맞은 날 {(~r.miss).sum()}), 하루 평균 헤드라인 {r.n_heads.mean():.0f}개\n")

base = r.miss.mean()
print(f"[이슈 범주별] 표본 틀린비율 기준 {base*100:.1f}%")
print(f"  {'이슈':10s} {'있는날':>5s} {'있을때 틀림':>10s} {'없을때 틀림':>10s} {'차이':>6s}")
res = []
for c in CAT:
    has = r[c] >= 2        # 헤드라인 2건 이상 = 그날의 이슈
    if has.sum() < 8 or (~has).sum() < 8: continue
    a, b = r.miss[has].mean(), r.miss[~has].mean(); res.append((c, has.sum(), a, b, a - b))
for c, n, a, b, df_ in sorted(res, key=lambda x: -x[4]):
    print(f"  {c:10s} {n:5d}  {a*100:9.1f}%  {b*100:9.1f}%  {df_*100:+5.1f}")

print("\n[뉴스 양] 헤드라인 수 많을수록?")
for lo, hi in ((0, 20), (20, 50), (50, 999)):
    m = (r.n_heads >= lo) & (r.n_heads < hi)
    if m.sum(): print(f"  {lo}~{hi}개: 틀림 {r.miss[m].mean()*100:.1f}% ({m.sum()}일)")

risky = [c for c, n, a, b, x in res if x >= .08]
if risky:
    m = (r[risky] >= 2).any(axis=1)
    print(f"\n[위험 이슈 {risky}] 있는 날 제외 시: 적중 {(~r.miss[~m]).mean()*100:.1f}% ({(~m).sum()}일)  "
          f"| 위험이슈 날 적중 {(~r.miss[m]).mean()*100:.1f}% ({m.sum()}일)  ※표본은 틀린날 비중이 높게 뽑혀 절대값보다 차이를 볼 것")

print("\n[최근 틀린 날 헤드라인 예시]")
for _, x in r[r.miss].tail(6).iterrows():
    tags = [c for c in CAT if x[c] >= 2]
    print(f"  {x.date} 이슈:{','.join(tags) or '-'}\n    " + " / ".join(x.heads.split(' || ')[:4])[:300])
