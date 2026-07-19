import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle

st.set_page_config(page_title="Gold Price Forecast", page_icon="🪙", layout="centered")

# ---------- Custom Styling ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
}
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(90deg, #f7d774, #e8b923);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
}
.subtitle {
    text-align: center;
    color: #b0b8c1;
    font-size: 15px;
    margin-bottom: 30px;
}
.price-card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(232, 185, 35, 0.3);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin-bottom: 25px;
    backdrop-filter: blur(10px);
}
.price-label {
    color: #b0b8c1;
    font-size: 14px;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.price-value {
    color: #f7d774;
    font-size: 48px;
    font-weight: 800;
    margin-top: 5px;
}
.price-date {
    color: #7a8494;
    font-size: 13px;
    margin-top: 5px;
}
.forecast-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
}
.forecast-title {
    color: #ffffff;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 12px;
}
.metric-row {
    display: flex;
    justify-content: space-between;
}
.metric-box {
    text-align: center;
    flex: 1;
}
.metric-label {
    color: #8a93a3;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-value {
    font-size: 22px;
    font-weight: 700;
    margin-top: 4px;
}
.low-val { color: #ff6b6b; }
.mid-val { color: #f7d774; }
.high-val { color: #51cf66; }
.footer-note {
    text-align: center;
    color: #6b7280;
    font-size: 12px;
    margin-top: 30px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🪙 Gold Price Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered 1–3 day gold price prediction using market indicators (DXY, Oil, US 10Y Yield, S&P500)</div>', unsafe_allow_html=True)


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
    latest_date = latest_row.index[0].strftime('%d %b %Y')

    # ---------- Current Price Card ----------
    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">Current Gold Price (USD / troy oz)</div>
        <div class="price-value">${current_price:,.2f}</div>
        <div class="price-date">As of {latest_date}</div>
    </div>
    """, unsafe_allow_html=True)

    # ---------- Forecast Cards ----------
    horizon_map = [('day1', 'Tomorrow'), ('day2', 'In 2 Days'), ('day3', 'In 3 Days')]

    for horizon, label in horizon_map:
        q_low = models[f'{horizon}_q10'].predict(X_latest)[0]
        q_mid = models[f'{horizon}_q50'].predict(X_latest)[0]
        q_high = models[f'{horizon}_q90'].predict(X_latest)[0]

        low_price = current_price * (1 + q_low)
        mid_price = current_price * (1 + q_mid)
        high_price = current_price * (1 + q_high)

        st.markdown(f"""
        <div class="forecast-card">
            <div class="forecast-title">📅 {label}</div>
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-label">Low</div>
                    <div class="metric-value low-val">${low_price:,.0f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Expected</div>
                    <div class="metric-value mid-val">${mid_price:,.0f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">High</div>
                    <div class="metric-value high-val">${high_price:,.0f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer-note">
        ⚠️ For informational and educational purposes only. Not financial advice.<br>
        Model: LightGBM Quantile Regression &nbsp;|&nbsp; Features: Gold, DXY, Oil, 10Y Yield, S&P500
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Something went wrong loading data or models: {e}")
