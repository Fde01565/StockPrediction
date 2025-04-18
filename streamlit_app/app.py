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

if st.button("Predict"):

    # Step 1: Get the stock's last 6 months of closing prices
    df = yf.download(ticker, period="6mo")
    close_prices = df[['Close']].values

    # Step 2: Scale prices between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Step 3: Take the last 30 days as input
    last_30_days = scaled_data[-30:]
    input_sequence = last_30_days.reshape(1, 30, 1)

    # Step 4: Make prediction
    predicted_scaled = model.predict(input_sequence)
    predicted_price = scaler.inverse_transform(predicted_scaled)

    # Step 5: Show prediction
    st.subheader("📅 Predicted Closing Price for Tomorrow:")
    st.success(f"${predicted_price[0][0]:.2f}")

    # Step 6: Plot past prices + predicted price line
    fig, ax = plt.subplots()
    ax.plot(df['Close'][-60:], label="Past 60 Days")
    ax.axhline(y=predicted_price[0][0], color='orange', linestyle='--', label="Predicted Price")
    ax.legend()
    st.pyplot(fig)
