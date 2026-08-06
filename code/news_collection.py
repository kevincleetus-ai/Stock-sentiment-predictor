import requests
import pandas as pd

API_KEY = "ddbe6c80440e482cbe37edbba3b709b4"
ticker = "AAPL"
company = "Apple"

# pull latest news articles mentioning the company
url = f"https://newsapi.org/v2/everything?q={company}+stock&language=en&sortBy=publishedAt&apiKey={API_KEY}"
response = requests.get(url)
data = response.json()
articles = data["articles"]

headlines = []

# grab just the fields we need
for article in articles:
    headlines.append({
        "date": article["publishedAt"],
        "headline": article["title"],
        "source": article["source"]["name"]
    })

df = pd.DataFrame(headlines)
df.to_csv("data/AAPL_news.csv", index=False)

print(f"collected {len(df)} headlines")
print(df.head())