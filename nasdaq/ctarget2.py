"""목표② 확장 탐색 + 최근 한 달 점검
- ①에서 쓰던 재료(이슈일·월초·VIX·아시아/유럽 되돌림·월요일 세분화·조합)를 ②에 적용
- 과최적화 방지: 앞 절반/뒤 절반 모두 기준선(항상매수)보다 좋아야 '후보'
- 후보 + 기존(월요일 매수, 모델 확신≥65%)을 최근 21거래일에 적용
"""
import io, re, contextlib
import numpy as np, pandas as pd, requests, yfinance as yf
with contextlib.redirect_stdout(io.StringIO()):
    import ctarget as c
D, up, Pm = c.D.copy(), c.up, c.Pm
R = D.C.values
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}

# 이슈일 (② 구간 안에서 터지는 것)
fomc = set()
try:
    h = requests.get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", headers=UA, timeout=30).text
    for yr, body in re.findall(r"(20\d\d) FOMC Meetings(.*?)(?=20\d\d FOMC Meetings|$)", h, re.S):
        for mon, day in re.findall(r'fomc-meeting__month[^>]*>\s*<strong>([A-Za-z/]+)</strong>.*?fomc-meeting__date[^>]*>([\d\-\*\s]+)<', body, re.S):
            try: fomc.add(pd.Timestamp(f"{mon.split('/')[-1][:3]} {re.findall(r'[0-9]+', day)[-1]} {yr}"))
            except Exception: pass
except Exception as e: print("FOMC 실패", e)
earn = {}
for t in ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AVGO"]:
    try:
        for ts in yf.Ticker(t).get_earnings_dates(limit=16).index:
            ts = pd.Timestamp(ts).tz_convert("America/New_York")
            if ts.hour >= 12: earn.setdefault(ts.normalize().tz_localize(None), []).append(t)   # 당일 장마감 후 발표 = 그날 밤 ② 구간
    except Exception: pass
idx = D.index
ev_fomc = idx.isin(fomc)
ev_earn = idx.isin(list(earn)); ev_nvda = np.array([("NVDA" in earn.get(x, [])) for x in idx])

ones = np.ones(len(D))
mon = D.mon.values == 1
rules = {
    "항상 매수": (ones, ones.astype(bool)),
    "월요일 매수": (ones, mon),
    "월요일+주말점프(갭)↑ 매수": (ones, mon & (D.gap.values > 0)),
    "월요일+갭↓ 매수": (ones, mon & (D.gap.values < 0)),
    "월요일+상승세 매수": (ones, mon & (D.reg.values == 1)),
    "월요일+하락세 매수": (ones, mon & (D.reg.values == -1)),
    "VIX≥20 매수": (ones, D.vix.values >= 20),
    "VIX≥25 매수": (ones, D.vix.values >= 25),
    "VIX<15 매도": (-ones, D.vix.values < 15),
    "닛케이 -0.5%↓ 매수": (ones, D.n225.values <= -.005),
    "닛케이 +0.5%↑ 매도": (-ones, D.n225.values >= .005),
    "DAX -0.5%↓ 매수": (ones, D.dax_gap.values <= -.005),
    "DAX +0.5%↑ 매도": (-ones, D.dax_gap.values >= .005),
    "아시아·유럽 모두 하락 매수": (ones, (D.n225.values < 0) & (D.dax_gap.values < 0)),
    "아시아·유럽 모두 상승 매도": (-ones, (D.n225.values > 0) & (D.dax_gap.values > 0)),
    "아시아·유럽 하락+VIX≥18 매수": (ones, (D.n225.values < 0) & (D.dax_gap.values < 0) & (D.vix.values >= 18)),
    "월초3일 매수": (ones, D.mstart.values == 1),
    "FOMC 결정일 밤 매수": (ones, ev_fomc),
    "빅테크 실적 발표 밤 매수": (ones, ev_earn),
    "엔비디아 실적 밤 매수": (ones, ev_nvda),
    "금요일 매도": (-ones, D.fri.values == 1),
}
half = len(D) // 2
print(f"[②] {idx[0].date()}~{idx[-1].date()} {len(D)}일. 기준선 항상매수 {up.mean()*100:.1f}% (앞 {up[:half].mean()*100:.1f} / 뒤 {up[half:].mean()*100:.1f})\n")
print(f"  {'규칙':26s} {'적중':>6s} {'일수':>4s} {'건당':>7s} | {'앞절반':>6s} {'뒤절반':>6s} 판정")
cands = {}
for nm, (s, m) in rules.items():
    if m.sum() < 8: print(f"  {nm:26s} (표본 {m.sum()}일 부족)"); continue
    hit = (np.sign(s) > 0) == up; r = np.sign(s) * R
    a, a1, a2 = hit[m].mean(), hit[m & (np.arange(len(D)) < half)].mean(), hit[m & (np.arange(len(D)) >= half)].mean()
    b1, b2 = (up[:half].mean(), up[half:].mean()) if s[0] > 0 else (1 - up[:half].mean(), 1 - up[half:].mean())
    good = a1 > max(.5, b1) + .03 and a2 > max(.5, b2) + .03 and r[m].mean() > 0 and nm != "항상 매수"
    if good: cands[nm] = (s, m)
    print(f"  {nm:26s} {a*100:5.1f}% {m.sum():4d} {r[m].mean()*100:+.3f}% | {a1*100:5.1f}% {a2*100:5.1f}% {'★후보' if good else ''}")

conf = np.abs(Pm - .5) + .5
cands_all = {"월요일 매수(기존)": (ones, mon), "모델 확신≥65%(기존)": (np.sign(Pm - .5), (~np.isnan(Pm)) & (conf >= .65))}
cands_all.update(cands)
last = np.arange(len(D)) >= len(D) - 21
print(f"\n[최근 한 달 {idx[last][0].date()} ~ {idx[-1].date()}, 21거래일] 기준선 항상매수 {up[last].mean()*100:.0f}% 건당 {R[last].mean()*100:+.3f}%")
for nm, (s, m) in cands_all.items():
    mm = m & last
    if not mm.sum(): print(f"  {nm:26s} 해당일 없음"); continue
    hit = (np.sign(s) > 0) == up
    print(f"  {nm:26s} {hit[mm].sum()}/{mm.sum()} = {hit[mm].mean()*100:3.0f}%  건당 {(np.sign(s)*R)[mm].mean()*100:+.3f}%  날짜: " +
          ", ".join(f"{x.strftime('%m/%d')}{'O' if h else 'X'}" for x, h in zip(idx[mm], hit[mm])))
# 후보 합치기(하나라도 해당하면 매수/매도, 충돌 시 쉼)
if len(cands_all) > 1:
    vote = np.zeros(len(D))
    for nm, (s, m) in cands_all.items(): vote += np.where(m, np.sign(s), 0)
    m = vote != 0; hit = (vote > 0) == up
    print(f"\n[후보 합본] 전체 {hit[m].mean()*100:.1f}% ({m.sum()}일, 건당 {(np.sign(vote)*R)[m].mean()*100:+.3f}%) | 최근 한 달 {hit[m & last].sum()}/{(m & last).sum()}")
