import requests
import pandas as pd
API_KEY = "ddbe6c80440e482cbe37edbba3b709b4"
ticker = "AAPL"
company = "Apple"
url = f"https://newsapi.org/v2/everything?q={company}&language=en&sortBy=publishedAt&apiKey={API_KEY}"
response = requests.get(url)
data = response.json()
articles = data["articles"]
headlines = []
for article in articles:
    headlines.append({
        "date": article["publishedAt"],
        "headline": article["title"],
        "source": article["source"]["name"]
    })
df = pd.DataFrame(headlines)
df.to_csv("data/AAPL_news.csv", index=False)
print(f"Collected {len(df)} headlines")
print(df.head())