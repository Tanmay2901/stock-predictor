import yfinance as yf
import pandas as pd

TICKER = "AAPL"
START = "2015-01-01"
END = "2025-01-01"

df = yf.download(
    TICKER,
    start=START,
    end=END,
    auto_adjust=False,
)

# yfinance may return two-level columns, even for one ticker.
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

out_path = f"data/{TICKER.lower()}.csv"
df.to_csv(out_path)

print(f"Saved {len(df)} rows to {out_path}")
print(df.columns.tolist())
print(df.head())
print(df.tail())