"""이슈(이벤트) 백테스트: 나스닥에 영향이 확인된 '예정된 이슈' 날짜를 모아
그날 나스닥 상황(상승률, 변동폭, 8시 갭 적중률)을 비교하고 오늘·내일 이슈를 확인한다.

이슈 목록 (발표 시각 KST, 겨울 기준)
  CPI·PPI·고용(NFP)  22:30  ← 8시 이후 발표
  FOMC 결정          04:00(다음날 새벽) ← 8시 이후
  빅테크 실적          전날 장마감 후 발표 → 8시 갭에 이미 반영
  옵션만기 / 쿼드러플위칭, 월초·월말
출처: BLS 보도자료 아카이브, 연준 FOMC 일정, yfinance 실적일
"""
import re
import numpy as np, pandas as pd, requests, yfinance as yf
from pattern import df, gap

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}
dates = pd.DatetimeIndex(df.date)
d = df.copy(); d.index = dates; d["gap"] = gap
d = d[d.y.notna()].copy(); d["y"] = d.y.astype(bool)
d["A"] = d.post.apply(lambda a: a[-1]); d["B"] = (1 + d.A) / (1 + d.gap) - 1
ev = {}


def bls(series):
    try:
        h = requests.get(f"https://www.bls.gov/bls/news-release/{series}.htm", headers=UA, timeout=30).text
        out = {pd.Timestamp(f"{m[4:]}-{m[:2]}-{m[2:4]}") for m in re.findall(rf"{series}_(\d{{8}})\.htm", h)}
        h2 = requests.get(f"https://www.bls.gov/schedule/news_release/{series}.htm", headers=UA, timeout=30).text
        for s in re.findall(r"([A-Z][a-z]{2})\.? (\d{1,2}), (20\d\d)", h2):
            try: out.add(pd.Timestamp(f"{s[0]} {s[1]} {s[2]}"))
            except Exception: pass
        return sorted(out)
    except Exception as e:
        print(f"  ! {series} 일정 수집 실패: {e}"); return []


for nm, s in (("CPI 발표일", "cpi"), ("PPI 발표일", "ppi"), ("고용지표(NFP)", "empsit")):
    ev[nm] = bls(s)

try:  # FOMC
    h = requests.get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", headers=UA, timeout=30).text
    fomc = []
    for yr, body in re.findall(r"(20\d\d) FOMC Meetings(.*?)(?=20\d\d FOMC Meetings|$)", h, re.S):
        for mon, day in re.findall(r'fomc-meeting__month[^>]*>\s*<strong>([A-Za-z/]+)</strong>.*?fomc-meeting__date[^>]*>([\d\-\*\s]+)<', body, re.S):
            mon = mon.split("/")[-1]; last = re.findall(r"\d+", day)[-1]
            try: fomc.append(pd.Timestamp(f"{mon[:3]} {last} {yr}"))
            except Exception: pass
    ev["FOMC 결정일(발표 당일)"] = sorted(set(fomc))
    ev["FOMC 다음날(결과 반영)"] = [x + pd.offsets.BDay(1) for x in ev["FOMC 결정일(발표 당일)"]]
except Exception as e:
    print(f"  ! FOMC 수집 실패: {e}")

big = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AVGO"]
earn = []
for t in big:
    try:
        e = yf.Ticker(t).get_earnings_dates(limit=16)
        for ts in e.index:
            ts = pd.Timestamp(ts).tz_convert("America/New_York")
            react = ts.normalize().tz_localize(None) + (pd.offsets.BDay(1) if ts.hour >= 12 else pd.Timedelta(0))
            earn.append((react, t))
    except Exception as ex:
        print(f"  ! {t} 실적일 실패: {ex}")
ev["빅테크 실적 반영일(전체)"] = sorted({x for x, _ in earn})
ev["엔비디아 실적 반영일"] = sorted({x for x, t in earn if t == "NVDA"})

fri3 = [x for x in pd.bdate_range("2024-01-01", "2027-12-31") if x.weekday() == 4 and 15 <= x.day <= 21]
ev["옵션만기(월간)"] = [x for x in fri3 if x.month not in (3, 6, 9, 12)]
ev["쿼드러플위칭"] = [x for x in fri3 if x.month in (3, 6, 9, 12)]

macro = set(ev.get("CPI 발표일", []) + ev.get("PPI 발표일", []) + ev.get("고용지표(NFP)", []) + ev.get("FOMC 결정일(발표 당일)", []))
ev["8시 이후 발표 이슈(합계)"] = sorted(macro)

# ---------------- 백테스트 ----------------
base_up = d.y.mean() * 100; base_abs = d.A.abs().mean() * 100
base_gap = ((d.gap > 0) == d.y).mean() * 100
base_B = ((d.B > 0) == (d.gap > 0)).mean() * 100
any_ev = set().union(*[set(v) for v in ev.values()])
no = d[~d.index.isin(any_ev)]
print(f"\n기간 {d.index[0].date()} ~ {d.index[-1].date()}  {len(d)}일")
print(f"{'이슈':22s} {'일수':>4s} {'상승%':>6s} {'평균변동':>7s} {'8시갭적중':>8s} {'8시후 갭방향':>10s}")
print(f"{'(전체 평균)':22s} {len(d):4d} {base_up:5.1f}% {base_abs:6.2f}% {base_gap:7.1f}% {base_B:9.1f}%")
print(f"{'(이슈 없는 날)':22s} {len(no):4d} {no.y.mean()*100:5.1f}% {no.A.abs().mean()*100:6.2f}% "
      f"{((no.gap>0)==no.y).mean()*100:7.1f}% {((no.B>0)==(no.gap>0)).mean()*100:9.1f}%")
for nm, ds in ev.items():
    s = d[d.index.isin(ds)]
    if len(s) < 3:
        print(f"{nm:22s} {len(s):4d}  (데이터 부족)"); continue
    print(f"{nm:22s} {len(s):4d} {s.y.mean()*100:5.1f}% {s.A.abs().mean()*100:6.2f}% "
          f"{((s.gap>0)==s.y).mean()*100:7.1f}% {((s.B>0)==(s.gap>0)).mean()*100:9.1f}%")

print("\n[8시 이후 발표 이슈 날, 갭 크기별 적중]")
s = d[d.index.isin(macro)]
for lo in (0, .0025, .005):
    m = s.gap.abs() >= lo
    print(f"  |갭|≥{lo*100:.2f}%: {((s.gap[m]>0)==s.y[m]).mean()*100:5.1f}% ({m.sum()}일)  vs 이슈없는날 "
          f"{((no.gap[no.gap.abs()>=lo]>0)==no.y[no.gap.abs()>=lo]).mean()*100:5.1f}%")

today = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
print("\n[오늘·향후 5거래일 이슈]")
for x in pd.bdate_range(today - pd.Timedelta(days=1), periods=6):
    hit = [nm for nm, ds in ev.items() if x in set(ds) and "합계" not in nm]
    print(f"  {x.date()} ({'월화수목금'[x.weekday()]}): {', '.join(hit) if hit else '-'}")
print("\n수집 건수: " + ", ".join(f"{k} {len([x for x in v if x >= d.index[0]])}" for k, v in ev.items()))
