import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="📊 Advanced Stock Predictor", layout="wide")

# Load pre-trained LSTM model
@st.cache_resource
def load_lstm_model():
    return load_model("model.h5")

model = load_lstm_model()

# Title and Description
st.title("📈 Advanced Stock Price Prediction App")
st.markdown("""
This application uses a Long Short-Term Memory (LSTM) neural network to predict future stock prices based on recent historical data.
You can visualize past trends, predict future values, and even download your prediction results.
""")

# Sidebar Inputs
st.sidebar.header("Stock Parameters")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL").upper()
forecast_days = st.sidebar.slider("Number of Days to Predict", 1, 30, 5)
data_period = st.sidebar.selectbox("Select Historical Period", ["3mo", "6mo", "1y", "2y", "5y"], index=1)
run_forecast = st.sidebar.button("📊 Run Forecast")

# Fetch stock data
def fetch_data(ticker, period):
    df = yf.download(ticker, period=period)
    df = df[["Close"]]
    df.dropna(inplace=True)
    return df

# Normalize and predict
def predict_prices(df, days):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    seq = scaled[-30:]
    predictions = []

    for _ in range(days):
        seq_input = seq.reshape(1, 30, 1)
        pred_scaled = model.predict(seq_input, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)
        predictions.append(pred[0][0])
        seq = np.append(seq, [[pred_scaled[0][0]]], axis=0)[-30:]

    return predictions, scaler.inverse_transform(scaled)

if run_forecast:
    st.subheader(f"🔍 Fetching data for **{ticker}**")
    df = fetch_data(ticker, data_period)

    if df.empty:
        st.error("Invalid ticker or no data available.")
    else:
        st.success("Data loaded successfully!")

        st.subheader("📉 Historical Closing Prices")
        st.line_chart(df)

        predictions, unscaled = predict_prices(df, forecast_days)

        # Future Dates
        future_dates = [df.index[-1] + timedelta(days=i + 1) for i in range(forecast_days)]

        # Display Table
        result_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Close Price ($)": predictions
        })
        st.subheader("🔮 Forecast Table")
        st.dataframe(result_df)

        # Plot Forecast
        st.subheader("📈 Forecast Chart")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index[-60:], df["Close"].values[-60:], label="Past 60 Days")
        ax.plot(future_dates, predictions, label="Forecast", linestyle="--", marker="o")
        ax.set_title(f"{ticker} Stock Price Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.legend()
        st.pyplot(fig)

        # Export CSV
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Forecast as CSV", csv, f"{ticker}_forecast.csv", "text/csv")
