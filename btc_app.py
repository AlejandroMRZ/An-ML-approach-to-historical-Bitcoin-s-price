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
@st.cache_data
def fetch_all_assets_data():
    # Fetch data for multiple assets using Yahoo Finance
    btc_data = yf.download('BTC-USD', start='2012-01-01', progress=False)
    gold_data = yf.download('GC=F', start='2012-01-01', progress=False)  # Gold Futures
    sp500_data = yf.download('^GSPC', start='2012-01-01', progress=False)  # S&P 500 Index
    tesla_data = yf.download('TSLA', start='2012-01-01', progress=False)  # Tesla Stock
    microstrategy_data = yf.download('MSTR', start='2012-01-01', progress=False)  # MicroStrategy Stock

    # Select closing prices and rename columns for clarity
    btc_data = btc_data[['Close']].rename(columns={'Close': 'Bitcoin'})
    gold_data = gold_data[['Close']].rename(columns={'Close': 'Gold'})
    sp500_data = sp500_data[['Close']].rename(columns={'Close': 'S&P 500'})
    tesla_data = tesla_data[['Close']].rename(columns={'Close': 'Tesla'})
    microstrategy_data = microstrategy_data[['Close']].rename(columns={'Close': 'MicroStrategy'})

    # Combine all data into a single DataFrame
    combined_df = btc_data.join(gold_data, how='outer').join(sp500_data, how='outer')
    combined_df = combined_df.join(tesla_data, how='outer').join(microstrategy_data, how='outer')

    # Normalize prices to the first available value (to start at 100%)
    normalized_df = combined_df / combined_df.iloc[0] * 100

    return normalized_df


if menu == "Analytics":
    st.title("Analytics & Insights")

    # Fetch data
    assets_data = fetch_all_assets_data()

    # Date filters
    start_date = st.date_input("Start Date", value=assets_data.index.min().date(), key="analytics_start")
    end_date = st.date_input("End Date", value=assets_data.index.max().date(), key="analytics_end")

    # Filter data for the selected date range
    filtered_assets_data = assets_data.loc[f"{start_date}":f"{end_date}"]

    if not filtered_assets_data.empty:
        st.subheader("Price Comparison (Logarithmic Scale)")

        # Ensure column names are strings (to avoid tuple issue)
        filtered_assets_data.columns = filtered_assets_data.columns.map(str)

        # Create a multi-line chart
        fig = go.Figure()

        # Add traces for each asset
        for column in filtered_assets_data.columns:
            fig.add_trace(go.Scatter(
                x=filtered_assets_data.index,
                y=filtered_assets_data[column],
                mode='lines',
                name=column  # Use the column name as a string
            ))

        # Customize layout with logarithmic scaling
        fig.update_layout(
            title="Price Comparison Since 2012 (Logarithmic Scale)",
            xaxis_title="Date",
            yaxis_title="Logarithmic Price Scale",
            yaxis_type="log",  # Logarithmic scale applied
            template="plotly_dark",
            legend_title="Assets"
        )

        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for the selected date range.")

    # Path to the GIF
    gif_path = "img/S5s.gif"

    # Fetch the Fear and Greed Index
    fear_and_greed = fetch_fear_and_greed_index()

    if fear_and_greed:
        # Display the index
        st.subheader("📈 Fear and Greed Index")
        st.write("The Fear and Greed Index indicates the current market sentiment for Bitcoin.")
        
        # Show the index value and description
        index_value = int(fear_and_greed['value'])
        sentiment = fear_and_greed['value_classification']

        # Customize colors for the sentiment
        sentiment_color = "green" if sentiment in ["Greed", "Extreme Greed"] else "red" if sentiment in ["Fear", "Extreme Fear"] else "orange"

        # Display sentiment in a styled container
        st.markdown(
            f"""
            <div style="padding: 10px; border-radius: 5px; background-color: #f5f5f5; border: 1px solid #ccc;">
                <h3 style="color: {sentiment_color}; margin: 0;">{sentiment} ({index_value})</h3>
                <p style="margin: 0;">As of {pd.to_datetime(fear_and_greed['timestamp'], unit='s').strftime('%Y-%m-%d')}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Gauge chart for Fear and Greed Index
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=index_value,
            delta={'reference': 50},  # Neutral level
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

        st.plotly_chart(fig_gauge, use_container_width=True)

    st.write("""
    ### Sentiment Classification:
    - **Extreme Fear (0-25)**: Investors are very fearful, potentially a buying opportunity.
    - **Fear (26-50)**: General fear in the market.
    - **Neutral (51)**: Balanced sentiment in the market.
    - **Greed (52-75)**: Growing market optimism.
    - **Extreme Greed (76-100)**: Investors are overly greedy, potentially a signal for a market correction.
    """)

    # Display the GIF
    st.image(gif_path, caption="Interactive Bitcoin Animation", use_container_width=True)

    # Footer
    st.markdown("### Empowering your crypto insights – from data to decisions. 🚀 Made with ❤️ and Streamlit. Explore responsibly!")
    st.markdown("""
    <style>
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
