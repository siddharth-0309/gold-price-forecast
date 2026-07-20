# 🪙 Gold Price Trend Forecast

**Live App:** [gold-price-forecast.streamlit.app](https://gold-price-forecast-bcehaikrf8b5a3nnducoff.streamlit.app/)

An end-to-end machine learning system that forecasts short-term gold price trends (1–3 day horizons) using LightGBM quantile regression. Instead of predicting a single price point, the model estimates a **range of likely outcomes**, giving a more honest picture of market uncertainty than a point forecast.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)

---

## 📌 Overview

Gold prices move on macro signals more than isolated technical patterns. This project models short-term % price movement using a set of correlated market indicators, rather than relying on gold's own price history alone.

## ⚙️ How It Works

- **Model:** LightGBM Quantile Regression (predicts multiple quantiles instead of one point estimate)
- **Targets:** % price change across **3 time horizons** (1-day, 2-day, 3-day)
- **Features:** Correlated market signals —
  - DXY (US Dollar Index)
  - Crude Oil prices
  - US 10-Year Treasury Yield
  - S&P 500 Index
- **Output:** A prediction range per horizon with ~63–65% empirical coverage — the actual price fell inside the predicted range roughly two-thirds of the time in backtesting.

## ✨ Features

- 📊 Interactive Streamlit dashboard with a dark, gold-themed UI
- 💱 USD / INR toggle for Indian users
- 📈 Multi-horizon forecast (1–3 days) with prediction ranges, not just point values
- 🧮 Quantile regression for genuine uncertainty estimation

## 🗂️ Project Structure
gold-price-forecast/
├── app.py # Streamlit application
├── gold_models.pkl # Trained LightGBM quantile models
├── feature_cols.pkl # Feature column reference for inference
├── requirements.txt # Python dependencies
└── README.md

## 🧠 Why Quantile Regression?

Point predictions ("gold will be at $X tomorrow") give false confidence — markets don't work that way. Quantile regression instead outputs a **range** at chosen confidence levels, which better reflects real uncertainty and is more useful for decision-making than a single number.

## ⚠️ Disclaimer

This project is for educational and portfolio purposes only. It is **not financial advice** — do not use it as the sole basis for any investment decision.
