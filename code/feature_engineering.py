import pandas as pd

# Load stock price data
prices_df = pd.read_csv("data/AAPL_prices.csv")

# Load sentiment data
sentiment_df = pd.read_csv("data/AAPL_sentiment.csv")

# Convert date columns to datetime
prices_df["Date"] = pd.to_datetime(prices_df["Date"], utc=True).dt.date
sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.date

# Map sentiment labels to numbers
sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
sentiment_df["sentiment_score"] = sentiment_df["sentiment"].map(sentiment_map)

# Group sentiment by date  average score per day
daily_sentiment = sentiment_df.groupby("date").agg(
    avg_sentiment=("sentiment_score", "mean"),
    num_headlines=("sentiment_score", "count")
).reset_index()

# Merge with price data
merged_df = pd.merge(prices_df, daily_sentiment, left_on="Date", right_on="date", how="left")
# Fill missing sentiment with (neutral)
merged_df["avg_sentiment"] = merged_df["avg_sentiment"].fillna(0)
merged_df["num_headlines"] = merged_df["num_headlines"].fillna(0)

# Create target column 1 if price went up, 0 if it went down
merged_df["target"] = (merged_df["Close"].shift(-1) > merged_df["Close"]).astype(int)
# Drop last row (no next day price)
merged_df = merged_df[:-1]

# Save to CSV
merged_df.to_csv("data/AAPL_features.csv", index=False)

print(f"Dataset created with {len(merged_df)} rows")
print(merged_df.head())