import yfinance as yf
import pandas as pd

ticker = "AAPL"

stock = yf.Ticker(ticker)
df = stock.history(period="3y")

# reset so date becomes a proper column instead of the index
df = df.reset_index()

df.to_csv("data/AAPL_prices.csv", index=False)

print(f"Downloaded {len(df)} rows of data for {ticker}")
print(df.head())