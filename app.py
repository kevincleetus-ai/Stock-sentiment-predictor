import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import requests
from transformers import pipeline
from tenacity import retry, stop_after_attempt, wait_fixed

# Page title
st.title("📈 Stock Sentiment Predictor")
st.write("Enter a stock ticker to see price data and news sentiment analysis.")

# NewsAPI key
NEWS_API_KEY = "ddbe6c80440e482cbe37edbba3b709b4"

# Load FinBERT once
@st.cache_resource
def load_finbert():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

finbert = load_finbert()

# Retry mechanism for stock data
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_stock(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    if df.empty:
        raise ValueError("Empty dataframe")
    return df.reset_index()

# User input
ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA, GOOGL)", value="AAPL")
company = st.text_input("Enter Company Name (e.g. Apple, Tesla, Google)", value="Apple")

if st.button("Analyze"):

    # Stock price chart
    with st.spinner("Fetching stock data..."):
        try:
            df = fetch_stock(ticker)

            st.subheader(f"{ticker} Stock Price - Last 1 Year")
            fig, ax = plt.subplots()
            ax.plot(df["Date"], df["Close"])
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            st.pyplot(fig)

            st.subheader("Raw Price Data")
            st.dataframe(df.tail(10))

        except Exception as e:
            st.warning("Could not fetch stock data after 3 attempts. Please try again in a moment.")
            st.stop()

    # News + sentiment
    with st.spinner("Fetching news and analyzing sentiment..."):
        url = f"https://newsapi.org/v2/everything?q={company}+stock&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        data = response.json()
        articles = data["articles"]

        results = []
        for article in articles[:20]:
            headline = article["title"]
            try:
                sentiment = finbert(headline[:512])[0]
                results.append({
                    "date": article["publishedAt"],
                    "headline": headline,
                    "source": article["source"]["name"],
                    "sentiment": sentiment["label"],
                    "score": round(sentiment["score"], 2)
                })
            except:
                continue

        sentiment_df = pd.DataFrame(results)

        st.subheader(f"News Sentiment for {company}")
        st.dataframe(sentiment_df)

        # Sentiment summary chart
        st.subheader("Sentiment Summary")
        counts = sentiment_df["sentiment"].value_counts()
        fig2, ax2 = plt.subplots()
        color_map = {"positive": "green", "neutral": "gray", "negative": "red"}
        colors = [color_map.get(label, "blue") for label in counts.index]
        ax2.bar(counts.index, counts.values, color=colors)
        ax2.set_ylabel("Number of Headlines")
        st.pyplot(fig2)

    # Prediction
    with st.spinner("Making prediction..."):
        import joblib
        import numpy as np

        model = joblib.load("models/xgb_model.pkl")

        latest = df.tail(60).copy()
        latest["ma_7"] = latest["Close"].rolling(window=7).mean()
        latest["ma_30"] = latest["Close"].rolling(window=30).mean()
        latest["volatility"] = latest["Close"].rolling(window=7).std()
        latest["price_change_pct"] = latest["Close"].pct_change() * 100
        latest["volume_change_pct"] = latest["Volume"].pct_change() * 100
        delta = latest["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        latest["rsi"] = 100 - (100 / (1 + rs))

        avg_sent = sentiment_df["sentiment"].map({"positive": 1, "neutral": 0, "negative": -1}).mean()
        latest["avg_sentiment"] = avg_sent
        latest["num_headlines"] = len(sentiment_df)
        latest = latest.dropna()

        features = ["Open", "High", "Low", "Close", "Volume", "avg_sentiment", "num_headlines", "price_change_pct", "ma_7", "ma_30", "volatility", "volume_change_pct", "rsi"]
        X_latest = latest[features].tail(1)

        prediction = model.predict(X_latest)[0]

        st.subheader("📊 Prediction")
        if prediction == 1:
            st.success(f"🟢 {ticker} is predicted to go UP tomorrow!")
        else:
            st.error(f"🔴 {ticker} is predicted to go DOWN tomorrow!")

    st.success("Analysis complete!")