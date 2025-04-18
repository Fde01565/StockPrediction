# 📈 Stock Market Prediction System

A smart, hybrid AI system that predicts stock market movements by integrating historical price trends with real-time financial news analysis. This project aims to provide more informed and adaptive trading recommendations compared to traditional models.

---

## 🔍 Overview

Unlike models that rely solely on technical indicators, our system dynamically adapts to market-moving news events to enhance prediction accuracy.

**Supported stocks**: Any publicly traded company (e.g., TSLA, AAPL, AMZN).  
**Input**: Stock ticker (e.g., `TSLA`)  
**Output**: Forecasted price movement & trading recommendation

---

## 🧠 System Architecture

The system follows a multi-model pipeline:

1. **Stock Data Collection**: Fetches OHLCV price data and computes technical indicators.
2. **LSTM Model**: Predicts future stock prices based on historical patterns.
3. **News Impact Model**: Analyzes real-time financial news and assigns impact scores.
4. **Meta-Learning Layer**: Combines predictions from LSTM and News Impact Model.
5. **Final Output**: Provides a recommendation (e.g., Buy/Hold/Sell) based on combined insights.

---

## 🗃️ Data Sources

### 📊 Historical Stock Data
- Yahoo Finance / Alpha Vantage / Quandl APIs
- Technical Indicators: SMA, EMA, RSI, MACD, Bollinger Bands

### 📰 Financial News Data
- NewsAPI.org / Google News / Yahoo Finance scraping
- Headlines, summaries, timestamps, macroeconomic news

---

## 🧩 Models Used

### 1️⃣ LSTM (Long Short-Term Memory)
- Predicts future stock prices based on time-series patterns.
- Input: Historical price & technical indicators

### 2️⃣ News Impact Model
- NLP-based classification of financial news
- Outputs a market impact score based on past similar events

### 3️⃣ Meta-Learning Model
- Combines predictions from both models for enhanced accuracy
- Learns optimal weighting between price trends and news impact

---

## 👥 Team

| Member   | Role                                |
|----------|-------------------------------------|
| Bassam   | LSTM Model Development (AI Lead)    |
| Samir    | News Impact Model (NLP & Data)      |
| Faisal   | News Impact Model (NLP & Data)      |

---

## 🚧 Project Status

✅ Project Planning & Dataset Gathering  
🟡 Data Preprocessing (In Progress)  
🔲 Model Training & Evaluation  
🔲 Final Integration & Testing  
🔲 Deployment (Optional)

---

## 📌 Goals

- Achieve better short-term stock predictions using real-time context
- Create a modular, reusable system for other stocks
- Compare performance vs. traditional price-only models

---

## 📂 Folder Structure (To Be Updated)

---

## 📢 Note

This project is strictly educational and experimental. It does **not constitute financial advice**.


