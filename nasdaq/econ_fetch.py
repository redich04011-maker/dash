"""우리 밤(d) 미국 날짜의 경제지표 일정 수집 (나스닥 경제달력, 1건 약 5초)
결과: exchange/from_research/econ_events.csv (d, 이벤트, 시각ET)"""
import time, requests, pandas as pd
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36", "Accept": "application/json"}
O = pd.read_csv("in/NIGHTS_시간외_우리성과_밤별.csv", dtype={"d": str})
dates = sorted(set(pd.to_datetime(O.d)) | set(pd.bdate_range("2026-10-30", "2026-11-20")))
KEY = {"CPI": ["CPI"], "PPI": ["PPI"], "NFP": ["Nonfarm Payrolls"], "FOMC": ["Fed Interest Rate Decision", "FOMC Statement"],
       "PCE": ["PCE Price"], "GDP": ["GDP"], "소매판매": ["Retail Sales"], "실업수당": ["Initial Jobless Claims"], "ISM": ["ISM"]}
rows = []
for n, d in enumerate(dates):
    for _ in range(2):
        try:
            r = requests.get("https://api.nasdaq.com/api/calendar/economicevents", params={"date": d.strftime("%Y-%m-%d")}, headers=UA, timeout=20)
            data = (r.json().get("data") or {}).get("rows") or []
            for x in data:
                if "United States" not in (x.get("country") or ""): continue
                nm = x.get("eventName") or ""
                for k, ws in KEY.items():
                    if any(w.lower() in nm.lower() for w in ws):
                        rows.append({"d": d.strftime("%Y%m%d"), "종류": k, "이벤트": nm, "시각ET": x.get("gmt") or x.get("time")})
            break
        except Exception:
            time.sleep(2)
    if n % 25 == 0: print(f"{n}/{len(dates)}", flush=True)
    time.sleep(0.2)
df = pd.DataFrame(rows).drop_duplicates()
df.to_csv("exchange/from_research/econ_events.csv", index=False, encoding="utf-8-sig")
print(df.groupby("종류").size())
