import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the trained LSTM model
model = load_model("model.h5")

st.title("📈 Stock Price Predictor")

# User input
ticker = st.text_input("Enter Stock Ticker:", "AAPL")
num_days = st.number_input("How many days to predict?", min_value=1, max_value=30, value=1, step=1)

if st.button("Predict"):
    # Step 1: Get the stock's last 6 months of closing prices
    df = yf.download(ticker, period="6mo")

    if df.empty:
        st.error("⚠️ Invalid or unknown ticker symbol. Please enter a valid one like AAPL or TSLA.")
    else:
        close_prices = df[['Close']].values

        # Step 2: Scale prices between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        # Step 3: Take the last 30 days as input
        last_30_days = scaled_data[-30:]
        input_sequence = last_30_days.copy()

        predicted_prices = []

        for _ in range(num_days):
            input_seq = input_sequence.reshape(1, 30, 1)
            pred_scaled = model.predict(input_seq)
            pred_price = scaler.inverse_transform(pred_scaled)
            predicted_prices.append(pred_price[0][0])
            
            # Update the sequence: drop first, add predicted
            next_scaled = pred_scaled[0][0]
            input_sequence = np.append(input_sequence, [[next_scaled]], axis=0)[-30:]

        # Step 4: Show predicted prices
        st.subheader(f"📅 Predicted Closing Prices for Next {num_days} Day(s):")
        for i, price in enumerate(predicted_prices, start=1):
            st.write(f"Day {i}: **${price:.2f}**")

        # Step 5: Plot past + predicted
        fig, ax = plt.subplots()
        past_days = df['Close'][-60:]
        future_days = pd.date_range(start=past_days.index[-1] + pd.Timedelta(days=1), periods=num_days)

        ax.plot(past_days.index, past_days.values, label="Past 60 Days")
        ax.plot(future_days, predicted_prices, color='orange', linestyle='--', label="Predicted Future")

        ax.set_title(f"{ticker.upper()} – Past 60 Days + {num_days}-Day Forecast")
        ax.legend()
        st.pyplot(fig)
