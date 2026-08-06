# Stock Sentiment Predictor

A project that tries to predict whether a stock will go up or down the next day by combining price data with sentiment from recent news headlines.

The idea is pretty simple — if the news around a stock is mostly positive, it might go up. If its mostly negative, it might go down. I wanted to see if I could build something that actually tests that theory.

## How it works

- Pulls historical stock price data using yfinance
- Fetches recent news headlines using NewsAPI
- Runs each headline through FinBERT, a transformer model trained on financial text, to label it as positive, negative or neutral
- Combines the sentiment scores with technical indicators like RSI, moving averages and volatility
- Trains an XGBoost model to predict next day price movement
- Shows everything on a Streamlit dashboard that works for any stock you type in

## Tech Stack

- Python
- FinBERT (HuggingFace Transformers)
- XGBoost / Scikit-learn
- yfinance
- NewsAPI
- Streamlit
- Pandas / NumPy

## Project Structure

stock-sentiment-predictor/
├── app.py
├── code/
│   ├── data_collection.py
│   ├── news_collection.py
│   ├── sentiment_analysis.py
│   ├── feature_engineering.py
│   └── model_training.py
├── data/
│   ├── AAPL_prices.csv
│   ├── AAPL_news.csv
│   ├── AAPL_sentiment.csv
│   └── AAPL_features.csv
├── models/
│   └── xgb_model.pkl
└── requirements.txt

## Live Demo

https://stock-sentiment-kevin.streamlit.app

## Notes

Accuracy is limited by the free NewsAPI tier which only gives headlines from the last 30 days. The stock data goes back 3 years so most days have no sentiment data attached. With a better news source the model would have a lot more to work with and accuracy would improve.