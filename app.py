import streamlit as st
import pandas as pd
import yfinance as yf
from transformers import pipeline
import matplotlib.pyplot as plt

# Page title
st.title("📈 Stock Sentiment Predictor")
st.write("Enter a stock ticker to see price data and news sentiment analysis.")

# User input
ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA, GOOGL)", value="AAPL")

if st.button("Analyze"):
    with st.spinner("Fetching stock data..."):
        # Pull stock data
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        df = df.reset_index()

        # Plot stock price
        st.subheader(f"{ticker} Stock Price - Last 1 Year")
        fig, ax = plt.subplots()
        ax.plot(df["Date"], df["Close"])
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        st.pyplot(fig)

        # Show raw data
        st.subheader("Raw Price Data")
        st.dataframe(df.tail(10))

    st.success("Done! Sentiment analysis coming soon.")