from transformers import pipeline
import pandas as pd

# Load FinBERT sentiment analysis pipeline
print("Loading FinBERT model...")
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Load your headlines
df = pd.read_csv("data/AAPL_news.csv")

# Run sentiment analysis on each headline
print("Analyzing sentiment...")
results = []

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
        print(f"Skipped a headline due to error: {e}")
        continue

# Convert to dataframe
sentiment_df = pd.DataFrame(results)

# Save to CSV
sentiment_df.to_csv("data/AAPL_sentiment.csv", index=False)

print(f"Done! Analyzed {len(sentiment_df)} headlines")

# Print cleanly without encoding issues
try:
    print(sentiment_df.head().to_string())
except UnicodeEncodeError:
    print(sentiment_df[["date", "sentiment", "score"]].head())