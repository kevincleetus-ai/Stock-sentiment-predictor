import pandas as pd

prices_df = pd.read_csv("data/AAPL_prices.csv")
sentiment_df = pd.read_csv("data/AAPL_sentiment.csv")

print("Price dates sample:")
print(prices_df["Date"].head())
print("\nSentiment dates sample:")
print(sentiment_df["date"].head())