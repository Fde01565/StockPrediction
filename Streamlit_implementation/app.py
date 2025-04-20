import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load model
model = load_model("model.h5")

# App title
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Predictor")
st.markdown("Predict future stock prices using an AI-powered LSTM model.")

# Sidebar input
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
    num_days = st.number_input("Days to Predict", min_value=1, max_value=30, value=1)
    run_prediction = st.button("🔮 Predict")

if run_prediction:
    with st.spinner("Fetching data and generating predictions..."):
        df = yf.download(ticker, period="6mo")

        if df.empty:
            st.error("⚠️ Invalid or unknown ticker symbol. Try again with something like AAPL or TSLA.")
        else:
            close_prices = df[['Close']].values

            # Normalize data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_prices)

            # Prepare input sequence
            input_sequence = scaled_data[-30:].copy()
            predicted_prices = []

            for _ in range(num_days):
                input_seq = input_sequence.reshape(1, 30, 1)
                pred_scaled = model.predict(input_seq, verbose=0)
                pred_price = scaler.inverse_transform(pred_scaled)
                predicted_prices.append(pred_price[0][0])
                input_sequence = np.append(input_sequence, [[pred_scaled[0][0]]], axis=0)[-30:]

            # Get company info
            info = yf.Ticker(ticker).info
            company_name = info.get("longName", "Unknown Company")
            st.subheader(f"📊 {company_name} ({ticker})")

            # Show metric for next-day prediction
            st.metric(label=f"Prediction for Day 1", value=f"${predicted_prices[0]:.2f}")

            # Tabs for forecast and plot
            tab1, tab2 = st.tabs(["📅 Forecast Table", "📈 Chart"])

            with tab1:
                forecast_df = pd.DataFrame({
                    "Day": [f"Day {i+1}" for i in range(num_days)],
                    "Predicted Price ($)": [f"{p:.2f}" for p in predicted_prices]
                })
                st.table(forecast_df)

            with tab2:
                fig, ax = plt.subplots()
                past_days = df['Close'][-60:]
                future_days = pd.date_range(start=past_days.index[-1] + pd.Timedelta(days=1), periods=num_days)

                ax.plot(past_days.index, past_days.values, label="Past 60 Days")
                ax.plot(future_days, predicted_prices, linestyle="--", color="orange", label="Predicted")

                ax.set_title(f"{ticker} – Past 60 Days + {num_days}-Day Forecast")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price ($)")
                ax.legend()
                st.pyplot(fig)

            st.success("✅ Prediction complete!")
