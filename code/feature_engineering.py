import pandas as pd
import numpy as np

# Load stock price data
prices_df = pd.read_csv("data/AAPL_prices.csv")

# Load sentiment data
sentiment_df = pd.read_csv("data/AAPL_sentiment.csv")

# Convert date columns to datetime
prices_df["Date"] = pd.to_datetime(prices_df["Date"], utc=True).dt.date
sentiment_df["date"] = pd.to_datetime(sentiment_df["date"], utc=True).dt.date
# Map sentiment labels to numbers
sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
sentiment_df["sentiment_score"] = sentiment_df["sentiment"].map(sentiment_map)

# Group sentiment by date
daily_sentiment = sentiment_df.groupby("date").agg(
    avg_sentiment=("sentiment_score", "mean"),
    num_headlines=("sentiment_score", "count")
).reset_index()

# Merge with price data
merged_df = pd.merge(prices_df, daily_sentiment, left_on="Date", right_on="date", how="left")

# Fill missing sentiment with 0
merged_df["avg_sentiment"] = merged_df["avg_sentiment"].fillna(0)
merged_df["num_headlines"] = merged_df["num_headlines"].fillna(0)

# Add technical indicators
# Price change %
merged_df["price_change_pct"] = merged_df["Close"].pct_change() * 100

# 7 and 30 day moving averages
merged_df["ma_7"] = merged_df["Close"].rolling(window=7).mean()
merged_df["ma_30"] = merged_df["Close"].rolling(window=30).mean()

# Volatility — standard deviation over 7 days
merged_df["volatility"] = merged_df["Close"].rolling(window=7).std()

# Volume change %
merged_df["volume_change_pct"] = merged_df["Volume"].pct_change() * 100

# RSI — Relative Strength Index
delta = merged_df["Close"].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
rs = gain / loss
merged_df["rsi"] = 100 - (100 / (1 + rs))

# Create target column
merged_df["target"] = (merged_df["Close"].shift(-1) > merged_df["Close"]).astype(int)

# Drop rows with NaN values from rolling calculations
merged_df = merged_df.dropna(subset=["ma_7", "ma_30", "volatility", "rsi", "price_change_pct", "volume_change_pct"])
# Save to CSV
merged_df.to_csv("data/AAPL_features.csv", index=False)

print(f"Dataset created with {len(merged_df)} rows")
print(merged_df.head())