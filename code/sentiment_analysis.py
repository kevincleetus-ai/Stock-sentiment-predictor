from transformers import pipeline
import pandas as pd

print("Loading FinBERT model...")
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Load your headlines
df = pd.read_csv("data/AAPL_news.csv")

print("Analyzing sentiment...")
results = []

for headline in df["headline"]:
    result = finbert(headline[:512])  
    results.append({
        "headline": headline,
        "sentiment": result[0]["label"],
        "score": result[0]["score"]
    })

sentiment_df = pd.DataFrame(results)

sentiment_df.to_csv("data/AAPL_sentiment.csv", index=False)

print(f"Done! Analyzed {len(sentiment_df)} headlines")
print(sentiment_df.head())