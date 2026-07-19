import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle

st.set_page_config(page_title="Gold Price Forecast", page_icon="🪙")
st.title("🪙 Gold Price Forecast")
st.caption("Next 1-3 day gold price prediction using market indicators (DXY, Oil, Yields, S&P500)")

@st.cache_resource
def load_models():
    with open('gold_models.pkl', 'rb') as f:
        models = pickle.load(f)
    with open('feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    return models, feature_cols

@st.cache_data(ttl=3600)
def fetch_latest_data():
    tickers = {'gold': 'GC=F', 'dxy': 'DX-Y.NYB', 'oil': 'CL=F', 'yield10y': '^TNX', 'sp500': '^GSPC'}
    data_dict = {}
    for name, ticker in tickers.items():
        df = yf.Ticker(ticker).history(period='3mo')['Close']
        df.index = pd.to_datetime(df.index.date)
        data_dict[name] = df
    combined = pd.DataFrame(data_dict).sort_index().ffill().dropna()
    return combined

def build_features(combined):
    df = combined.copy()
    for lag in [1, 3, 5, 7]:
        df[f'gold_pct_lag_{lag}'] = df['gold'].pct_change(lag)
    df['gold_pct_change_1d'] = df['gold'].pct_change(1)
    df['gold_roll_mean_7'] = df['gold_pct_change_1d'].rolling(7).mean()
    df['gold_roll_std_7'] = df['gold_pct_change_1d'].rolling(7).std()
    for col in ['dxy', 'oil', 'yield10y', 'sp500']:
        df[f'{col}_pct_lag_1'] = df[col].pct_change(1).shift(1)
    df['day_of_week'] = df.index.dayofweek
    df['current_gold_price'] = df['gold']
    return df.dropna()

try:
    models, feature_cols = load_models()
    combined = fetch_latest_data()
    df_features = build_features(combined)

    latest_row = df_features.iloc[[-1]]
    current_price = latest_row['current_gold_price'].values[0]
    X_latest = latest_row[feature_cols]

    st.metric("Current Gold Price (USD/oz)", f"${current_price:,.2f}",
              help=f"As of {latest_row.index[0].strftime('%d %b %Y')}")

    st.divider()

    for horizon, label in [('day1', 'Tomorrow'), ('day2', 'In 2 Days'), ('day3', 'In 3 Days')]:
        q_low = models[f'{horizon}_q5'].predict(X_latest)[0]
        q_mid = models[f'{horizon}_q50'].predict(X_latest)[0]
        q_high = models[f'{horizon}_q95'].predict(X_latest)[0]

        low_price = current_price * (1 + q_low)
        mid_price = current_price * (1 + q_mid)
        high_price = current_price * (1 + q_high)

        st.subheader(label)
        col1, col2, col3 = st.columns(3)
        col1.metric("Low", f"${low_price:,.0f}")
        col2.metric("Expected", f"${mid_price:,.0f}")
        col3.metric("High", f"${high_price:,.0f}")

    st.divider()
    st.caption("⚠️ For informational/educational purposes only. Not financial advice.")
    st.caption("Model: LightGBM Quantile Regression | Features: Gold, DXY, Oil, 10Y Yield, S&P500")

except Exception as e:
    st.error(f"Something went wrong loading data or models: {e}")
