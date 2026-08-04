from transformers import pipeline
import pandas as pd

# takes a while to load so good to know when it starts
print("Loading FinBERT...")
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

df = pd.read_csv("data/AAPL_news.csv")

results = []

# run each headline through finbert, skip any that cause issues
for _, row in df.iterrows():
    try:
        result = finbert(row["headline"][:512])
        results.append({
            "date": row["date"],
            "headline": row["headline"],
            "sentiment": result[0]["label"],
            "score": result[0]["score"]
        })
    except Exception as e:
        print(f"skipped one: {e}")
        continue

sentiment_df = pd.DataFrame(results)

sentiment_df.to_csv("data/AAPL_sentiment.csv", index=False)

print(f"done — {len(sentiment_df)} headlines analyzed")

# some headlines have weird characters that break the terminal print
try:
    print(sentiment_df.head().to_string())
except UnicodeEncodeError:
    print(sentiment_df[["date", "sentiment", "score"]].head())