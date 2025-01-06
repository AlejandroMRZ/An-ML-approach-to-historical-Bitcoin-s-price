import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from binance.client import Client
import tensorflow as tf
import os
import yfinance as yf

# Set custom CSS for background color
st.markdown(
    """
    <style>
    /* Gradient background for the entire app */
    .stApp {
        background: linear-gradient(135deg, #D0EFFF, #FFFFFF);
    }
    </style>
    """,
    unsafe_allow_html=True
)

import requests

# Function to fetch the Fear and Greed Index
@st.cache_data
def fetch_fear_and_greed_index():
    url = "https://api.alternative.me/fng/?limit=1"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['data'][0]  # Get the latest Fear and Greed Index
    else:
        st.warning("Unable to fetch the Fear and Greed Index. Please try again later.")
        return None


# Function to fetch today's and yesterday's BTC prices
def fetch_btc_price():
    api_key = os.getenv("BINANCE_API_KEY")  # Add your Binance API Key as an environment variable
    api_secret = os.getenv("BINANCE_API_SECRET")  # Add your Binance API Secret as an environment variable

    if not api_key or not api_secret:
        st.warning("Binance API credentials are not set. Real-time price is unavailable.")
        return None, None

    client = Client(api_key, api_secret)

    # Fetch current price
    ticker = client.get_symbol_ticker(symbol="BTCUSDT")
    current_price = float(ticker["price"])

    # Fetch historical price (yesterday's close)
    klines = client.get_klines(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1DAY, limit=2)
    yesterday_close = float(klines[0][4])  # Close price of the previous day

    return current_price, yesterday_close



# Function to fetch Binance API data
@st.cache_data
def fetch_binance_data():
    api_key = os.getenv("BINANCE_API_KEY")  # Add your Binance API Key as an environment variable
    api_secret = os.getenv("BINANCE_API_SECRET")  # Add your Binance API Secret as an environment variable

    if not api_key or not api_secret:
        st.warning("Binance API credentials are not set. Real-time data is disabled.")
        return None

    client = Client(api_key, api_secret)
    klines = client.get_klines(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1DAY)

    # Transform API response into DataFrame
    binance_df = pd.DataFrame(klines, columns=[
        "Open Time", "Open", "High", "Low", "Close", "Volume", "Close Time", "Quote Asset Volume",
        "Number of Trades", "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"])

    binance_df['Timestamp'] = pd.to_datetime(binance_df['Open Time'], unit='ms')
    binance_df.set_index('Timestamp', inplace=True)
    
    # Use Quote Asset Volume for USD-based trading volume
    binance_df = binance_df[['Close', 'Quote Asset Volume']].rename(columns={'Quote Asset Volume': 'Volume (USD)'})
    binance_df = binance_df.astype(float)

    return binance_df


@st.cache_data
def load_data():
    # Load historical dataset
    df = pd.read_csv("btcusd_1-min_data.csv")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.set_index('Timestamp', inplace=True)
    df_daily = df.resample('D').mean()

    # Fetch and append real-time data
    binance_data = fetch_binance_data()
    if binance_data is not None:
        df_daily = pd.concat([df_daily, binance_data]).drop_duplicates()
        df_daily.sort_index(inplace=True)

    # Calculate moving averages
    df_daily['50_Day_MA'] = df_daily['Close'].rolling(window=50).mean()
    df_daily['200_Day_MA'] = df_daily['Close'].rolling(window=200).mean()

    return df_daily

# Function to identify golden and death crosses
def identify_crosses(df):
    df['Golden_Cross'] = (df['50_Day_MA'] > df['200_Day_MA']) & (df['50_Day_MA'].shift(1) <= df['200_Day_MA'].shift(1))
    df['Death_Cross'] = (df['50_Day_MA'] < df['200_Day_MA']) & (df['50_Day_MA'].shift(1) >= df['200_Day_MA'].shift(1))
    return df

# Load data
df = load_data()
df = identify_crosses(df)

# Load the pre-trained LSTM model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("lstm_btc_close_model.h5")
    return model

model = load_model()

# Predict BTC price for the next day
def predict_next_day(model, last_30_days):
    scaled_data = np.array(last_30_days).reshape(1, -1, 1)
    prediction_scaled = model.predict(scaled_data)[0, 0]
    return prediction_scaled

# Sidebar for navigation with expander
with st.sidebar.expander("Navigation", expanded=True):
    menu = st.radio("Navigate to:", ["Home", "Dashboard", "Analytics"])

# Home Section
if menu == "Home":
    st.title("Unlock the Power of Bitcoin Insights 🚀")
    st.write("""
    ### Dive deep into Bitcoin's past, uncover trends, and shape your strategy with our interactive analytics and dashboards.
    
    **Features include:**
    - Visualization of Bitcoin closing prices and moving averages.
    - Identification of golden and death crosses.
    - Real-time data updates from Binance API.
    - Next-day price prediction using LSTM model.
    
    Navigate to the **Dashboard** or **Analytics** sections using the sidebar.
    """)

    # Display today's BTC price with trend
    st.markdown("### Today's Bitcoin Price (USD)")
    btc_price, yesterday_price = fetch_btc_price()
    if btc_price and yesterday_price:
        # Determine trend
        if btc_price > yesterday_price:
            trend = "🔺 Up"
            trend_color = "green"
        elif btc_price < yesterday_price:
            trend = "🔻 Down"
            trend_color = "red"
        else:
            trend = "➡️ No Change"
            trend_color = "gray"

        # Display the price and trend
        st.markdown(
            f"<h3 style='color:{trend_color};'>💰 ${btc_price:,.2f} ({trend})</h3>",
            unsafe_allow_html=True
        )
    else:
        st.error("Unable to fetch Bitcoin price. Please check API credentials.")

    st.image("img/bitcoin.png", use_container_width=True)


# Dashboard Section
elif menu == "Dashboard":
    st.title("Bitcoin Price Dashboard")

    # Date filters
    start_date = st.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-31"))

    # Filter data
    filtered_df = df.loc[f"{start_date}":f"{end_date}"]

    if not filtered_df.empty:
        st.subheader(f"BTC Price & Technical Indicators: {start_date} to {end_date}")
        fig = go.Figure()

        # Add Close Price
        fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Close'], name='Close Price', line=dict(color='gold', width=2)))

        # Add Moving Averages
        fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['50_Day_MA'], name='50-Day MA', line=dict(color='blue', dash='dash')))
        fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['200_Day_MA'], name='200-Day MA', line=dict(color='red', dash='dash')))

        # Highlight Golden and Death Crosses
        golden_cross_dates = filtered_df[filtered_df['Golden_Cross']].index
        death_cross_dates = filtered_df[filtered_df['Death_Cross']].index

        # Add Golden Cross markers
        fig.add_trace(go.Scatter(
        x=golden_cross_dates,
        y=filtered_df.loc[golden_cross_dates, '50_Day_MA'],
        mode='markers',
        name='Golden Cross',
        marker=dict(color='green', size=10, symbol='x')
    ))

        # Add Death Cross markers
        fig.add_trace(go.Scatter(
            x=death_cross_dates,
            y=filtered_df.loc[death_cross_dates, '50_Day_MA'],
            mode='markers',
            name='Death Cross',
            marker=dict(color='red', size=10, symbol='x')
        ))

        # Update layout and display the figure
        fig.update_layout(
            title="Bitcoin Price with Moving Averages & Crosses",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cross Analysis Section
        st.write("### Cross Analysis")
        st.write("- **Golden Cross**: 50-Day MA crosses above 200-Day MA, indicating potential bullish momentum.")
        st.write("- **Death Cross**: 50-Day MA crosses below 200-Day MA, indicating potential bearish momentum.")

     

        # Predict Next Day Price
        st.subheader("Next Day BTC Price Prediction")
        last_30_days = filtered_df['Close'].dropna()[-30:].values
        if len(last_30_days) == 30:
            if st.button("Predict Price for Next Day"):
                scaled_last_30_days = (last_30_days - last_30_days.min()) / (last_30_days.max() - last_30_days.min())
                next_day_price_scaled = predict_next_day(model, scaled_last_30_days)
                next_day_price = next_day_price_scaled * (last_30_days.max() - last_30_days.min()) + last_30_days.min()
                st.success(f"Predicted BTC Price for the next day: ${next_day_price:.2f}")
        else:
            st.warning("Not enough data to make a prediction. Ensure 30 days of BTC data is available.")
    else:
        st.warning("No data available for the selected date range.")
    st.image("img/BTC.png", use_container_width=True)
        

# Analytics Section
st.title("Analytics & Insights")
@st.cache_data
def fetch_fear_and_greed_index():
    """Fetch the latest Fear and Greed Index."""
    url = "https://api.alternative.me/fng/?limit=1"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['data'][0]
    else:
        st.warning("Unable to fetch the Fear and Greed Index. Please try again later.")
        return None

@st.cache_data
def fetch_btc_data():
    """Fetch Bitcoin historical price data."""
    btc_data = yf.download('BTC-USD', start='2012-01-01', progress=False)['Close']
    return btc_data

def calculate_rsi(data, window=14):
    """Calculate the Relative Strength Index (RSI)."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Fetch Bitcoin data
btc_data = fetch_btc_data()

# Date Filters
start_date = st.date_input("Start Date", value=btc_data.index.min().date())
end_date = st.date_input("End Date", value=btc_data.index.max().date())

# Filter data
filtered_btc_data = btc_data.loc[f"{start_date}":f"{end_date}"]

if not filtered_btc_data.empty:
    # 1. Fear and Greed Index
    st.subheader("📈 Fear and Greed Index")
    fear_and_greed = fetch_fear_and_greed_index()
    if fear_and_greed:
        index_value = int(fear_and_greed['value'])
        sentiment = fear_and_greed['value_classification']
        sentiment_color = "green" if sentiment in ["Greed", "Extreme Greed"] else "red" if sentiment in ["Fear", "Extreme Fear"] else "orange"

        # Display the Fear and Greed Index
        st.markdown(
            f"""
            <div style="padding: 10px; border-radius: 5px; background-color: #f5f5f5; border: 1px solid #ccc;">
                <h3 style="color: {sentiment_color}; margin: 0;">{sentiment} ({index_value})</h3>
                <p style="margin: 0;">As of {pd.to_datetime(fear_and_greed['timestamp'], unit='s').strftime('%Y-%m-%d')}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Gauge Chart for Fear and Greed Index
        fig_fear_greed = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=index_value,
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': sentiment_color},
                'steps': [
                    {'range': [0, 25], 'color': 'red'},
                    {'range': [25, 50], 'color': 'orange'},
                    {'range': [50, 75], 'color': 'lightgreen'},
                    {'range': [75, 100], 'color': 'green'}
                ],
                'threshold': {
                    'line': {'color': sentiment_color, 'width': 4},
                    'thickness': 0.75,
                    'value': index_value
                }
            },
            title={'text': "Fear and Greed Index"}
        ))
        st.plotly_chart(fig_fear_greed, use_container_width=True)

import plotly.graph_objects as go

import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Function to calculate RSI
def calculate_rsi(close_prices, n=14):
    close_delta = close_prices.diff()

    # Separate positive and negative gains
    up_moves = close_delta.clip(lower=0)  # Keep only positive values
    down_moves = -close_delta.clip(upper=0)  # Keep only negative values as positive

    # Calculate moving averages
    avg_u = up_moves.rolling(window=n, min_periods=1).mean()
    avg_d = down_moves.rolling(window=n, min_periods=1).mean()

    # Calculate the Relative Strength (RS) and RSI
    rs = avg_u / avg_d
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Load the BTC dataset
btc_data = pd.read_csv("btcusd_1-min_data.csv")

# Preprocess the data
btc_data['Timestamp'] = pd.to_datetime(btc_data['Timestamp'], unit='s')
btc_data.set_index('Timestamp', inplace=True)

# Resample the data to monthly frequency (average closing prices per month)
btc_monthly = btc_data.resample('M').mean()

# Start and end dates for filtering
start_date = st.date_input("Start Date", value=btc_monthly.index.min().date(), key="rsi_start_date")
end_date = st.date_input("End Date", value=btc_monthly.index.max().date(), key="rsi_end_date")

# Filter the data for the selected date range
filtered_btc_data = btc_monthly.loc[start_date:end_date]

if not filtered_btc_data.empty:
    close_prices = filtered_btc_data['Close']
    rsi = calculate_rsi(close_prices)

    # Plot RSI with gradient color
    if not rsi.dropna().empty:
        fig_rsi = go.Figure()

        # Add RSI as scatter points with gradient color
        fig_rsi.add_trace(go.Scatter(
            x=filtered_btc_data.index,
            y=rsi,
            mode='markers+lines',
            name='RSI',
            marker=dict(
                size=8,
                color=rsi,  # Use RSI values for color
                colorscale='RdYlGn',  # Red-Yellow-Green color scale
                colorbar=dict(title="RSI")
            ),
            line=dict(color='gray', width=1)  # Add connecting lines in gray
        ))

        # Update layout
        fig_rsi.update_layout(
            title="Bitcoin Relative Strength Index (RSI) - Monthly Data",
            xaxis=dict(title="Date"),
            yaxis=dict(title="RSI", range=[0, 100]),
            template="plotly_dark"
        )

        # Display the chart
        st.plotly_chart(fig_rsi, use_container_width=True)

        # RSI Insights
        st.write("""
        ### RSI Insights:
        - **RSI > 70**: Bitcoin is overbought, signaling potential selling opportunities.
        - **RSI < 30**: Bitcoin is oversold, signaling potential buying opportunities.
        - Use RSI alongside other indicators to make informed decisions.
        """)
    else:
        st.warning("Not enough data to plot RSI.")
else:
    st.warning("No data available for the selected date range.")
