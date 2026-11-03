import pandas as pd, yfinance as yf
c = yf.download("NQ=F", period="1y", interval="1d", auto_adjust=True, progress=False)["Close"].squeeze()
c.index = pd.to_datetime(c.index).tz_localize(None)
d = pd.DataFrame({"종가": c, "MA5": c.rolling(5).mean(), "MA20": c.rolling(20).mean(),
                  "MA60": c.rolling(60).mean(), "MA120": c.rolling(120).mean()})
d["MA20기울기5일%"] = d.MA20.pct_change(5) * 100
d["20일등락%"] = c.pct_change(20) * 100
d["60일등락%"] = c.pct_change(60) * 100
d["고점대비%"] = (c / c.rolling(60).max() - 1) * 100
pd.set_option("display.width", 200)
print(d.tail(15).round(1).to_string())
