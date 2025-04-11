import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
import os
from datetime import datetime
import json

# Set up logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = os.path.join(log_dir, f"stock_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Application starting up")

# Configure session with retry strategy
logger.debug("Configuring request session with retry strategy")
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)

# Alpha Vantage API Configuration
ALPHA_VANTAGE_API_KEY = "5WF4COK1UFPHRLIO"  # Replace with your actual API key
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# Set page config
logger.debug("Setting up Streamlit page configuration")
st.set_page_config(
    page_title="Real-Time Stock Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
.main, .stApp {background-color: #0e1117; color: #fafafa;}
.stHeader {background-color: #262730;}
.stButton>button {background-color: #4CAF50; color: white; border-radius: 4px;}
.stSelectbox>div>div {background-color: #262730; color: white;}
.ticker-data {background-color: #262730; padding: 15px; border-radius: 5px; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📊 Real-Time Stock Market Tracker")
st.markdown("""
This app retrieves real-time stock price data and visualizes it with interactive charts.
Select a stock symbol, time period, and interval to get started!

> **Note:** This app uses Alpha Vantage's API which has rate limits.
> Free API keys are limited to 5 API requests per minute and 500 requests per day.
""")

# Sidebar for inputs
with st.sidebar:
    st.header("Chart Settings")
    
    # API Key Input
    user_api_key = st.text_input("Alpha Vantage API Key:", value=ALPHA_VANTAGE_API_KEY, 
                                 help="Get a free API key from https://www.alphavantage.co/support/#api-key")
    
    if user_api_key and user_api_key != "YOUR_ALPHA_VANTAGE_API_KEY":
        ALPHA_VANTAGE_API_KEY = user_api_key
    
    if ALPHA_VANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_API_KEY":
        st.warning("Please enter your Alpha Vantage API key")
    
    # Stock input (with default suggestions)
    stock_input = st.text_input("Enter Stock Symbol:", "AAPL").upper()
    logger.info(f"Stock input set to: {stock_input}")
    
    # Popular stock suggestions
    popular_stocks = st.multiselect(
        "Popular Stocks:",
        ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"],
        []
    )
    
    if popular_stocks:
        if len(popular_stocks) == 1:
            stock_input = popular_stocks[0]
        else:
            stock_input = ",".join(popular_stocks)
        logger.info(f"Stock selection updated via popular stocks: {stock_input}")
    
    # Time period selection (adapted for Alpha Vantage)
    output_size_options = {
        "Compact (Latest 100 data points)": "compact",
        "Full (All available data)": "full"
    }
    
    output_size = st.selectbox(
        "Select Data Size:",
        list(output_size_options.keys()),
        index=0
    )
    
    # Function/time series type selection
    function_options = {
        "Intraday": "TIME_SERIES_INTRADAY",
        "Daily": "TIME_SERIES_DAILY",
        "Weekly": "TIME_SERIES_WEEKLY",
        "Monthly": "TIME_SERIES_MONTHLY"
    }
    
    function = st.selectbox(
        "Select Time Series:",
        list(function_options.keys()),
        index=1  # Default to Daily
    )
    
    # Interval selection (only for intraday)
    interval_options = {
        "1 Minute": "1min",
        "5 Minutes": "5min",
        "15 Minutes": "15min",
        "30 Minutes": "30min",
        "60 Minutes": "60min"
    }
    
    if function == "Intraday":
        interval = st.selectbox(
            "Select Interval:",
            list(interval_options.keys()),
            index=2  # Default to 15min
        )
    else:
        interval = None
    
    # Chart type
    chart_type = st.selectbox(
        "Select Chart Type:",
        ["Candlestick", "Line", "OHLC"],
        index=0
    )
    logger.info(f"Selected chart type: {chart_type}")
    
    # Technical indicators
    st.subheader("Technical Indicators")
    show_ma = st.checkbox("Moving Averages", value=True)
    if show_ma:
        ma_periods = st.multiselect(
            "MA Periods:",
            [9, 20, 50, 100, 200],
            default=[20, 50]
        )
        logger.debug(f"Selected MA periods: {ma_periods}")
    
    show_volume = st.checkbox("Show Volume", value=True)
    show_bbands = st.checkbox("Bollinger Bands", value=False)
    
    # Add "Force Refresh" button
    if st.button("🔄 Force Refresh Data"):
        st.session_state.force_refresh = True
    else:
        if "force_refresh" not in st.session_state:
            st.session_state.force_refresh = False
            
    # Auto refresh
    st.subheader("Auto Refresh")
    enable_auto_refresh = st.checkbox("Enable Auto Refresh", value=False)
    
    if enable_auto_refresh:
        refresh_interval = st.slider(
            "Refresh Interval (seconds):",
            min_value=60,
            max_value=300,
            value=60,
            step=30,
            help="Alpha Vantage API has rate limits (5 calls/min for free tier)"
        )
        logger.info(f"Auto refresh enabled with interval: {refresh_interval}s")


# Helper functions
def get_alpha_vantage_data(symbol, function, output_size="compact", interval=None):
    """Fetch stock data from Alpha Vantage API"""
    logger.info(f"Fetching Alpha Vantage data for {symbol} (function: {function})")
    
    params = {
        "function": function,
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY,
        "outputsize": output_size,
        "datatype": "json"
    }
    
    if interval and function == "TIME_SERIES_INTRADAY":
        params["interval"] = interval
    
    try:
        logger.debug(f"Making Alpha Vantage API request with params: {params}")
        response = session.get(ALPHA_VANTAGE_BASE_URL, params=params)
        response.raise_for_status()
        
        data = response.json()
        logger.debug(f"Response keys: {list(data.keys())}")
        
        # Check for error messages
        if "Error Message" in data:
            logger.error(f"Alpha Vantage API error: {data['Error Message']}")
            return None, f"API Error: {data['Error Message']}"
        
        if "Note" in data:
            logger.warning(f"Alpha Vantage API note: {data['Note']}")
            st.warning(f"API Note: {data['Note']}")
        
        # Get the time series data key based on the function
        time_series_key = ""
        if function == "TIME_SERIES_INTRADAY":
            time_series_key = f"Time Series ({interval})"
        elif function == "TIME_SERIES_DAILY":
            time_series_key = "Time Series (Daily)"
        elif function == "TIME_SERIES_WEEKLY":
            time_series_key = "Weekly Time Series"
        elif function == "TIME_SERIES_MONTHLY":
            time_series_key = "Monthly Time Series"
        
        # Fallback to daily data if intraday is unavailable
        if time_series_key not in data and function == "TIME_SERIES_INTRADAY":
            logger.warning("Intraday data unavailable, falling back to daily data")
            params["function"] = "TIME_SERIES_DAILY"
            if "interval" in params:
                del params["interval"]
            response = session.get(ALPHA_VANTAGE_BASE_URL, params=params)
            data = response.json()
            time_series_key = "Time Series (Daily)"
        
        if time_series_key not in data:
            logger.error(f"Expected time series key '{time_series_key}' not found in response")
            logger.debug(f"Response content: {json.dumps(data)[:500]}...")
            return None, f"Expected data key '{time_series_key}' not found in API response"
        
        # Convert to DataFrame
        time_series_data = data[time_series_key]
        df = pd.DataFrame.from_dict(time_series_data, orient="index")
        
        # Rename columns from Alpha Vantage format to our expected format
        df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume"
        }, inplace=True)
        
        # Convert string values to float
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        
        # Sort by date (ascending)
        df.sort_index(inplace=True)
        
        # Get company info if available (from Global Quote endpoint)
        company_info = {}
        try:
            quote_params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": ALPHA_VANTAGE_API_KEY
            }
            quote_response = session.get(ALPHA_VANTAGE_BASE_URL, params=quote_params)
            quote_data = quote_response.json()
            
            if "Global Quote" in quote_data:
                quote = quote_data["Global Quote"]
                company_info["currentPrice"] = float(quote.get("05. price", 0))
                company_info["change"] = float(quote.get("09. change", 0))
                company_info["changePercent"] = quote.get("10. change percent", "0%")
                company_info["shortName"] = symbol
            
            # Try to get company overview for more data
            overview_params = {
                "function": "OVERVIEW",
                "symbol": symbol,
                "apikey": ALPHA_VANTAGE_API_KEY
            }
            overview_response = session.get(ALPHA_VANTAGE_BASE_URL, params=overview_params)
            overview_data = overview_response.json()
            
            if "Symbol" in overview_data:
                company_info.update({
                    "shortName": overview_data.get("Name", symbol),
                    "sector": overview_data.get("Sector", "N/A"),
                    "industry": overview_data.get("Industry", "N/A"),
                    "marketCap": float(overview_data.get("MarketCapitalization", 0)),
                    "beta": float(overview_data.get("Beta", 0)),
                    "trailingPE": float(overview_data.get("TrailingPE", 0)) if overview_data.get("TrailingPE") else "N/A",
                    "dividendYield": float(overview_data.get("DividendYield", 0)) if overview_data.get("DividendYield") else "N/A",
                    "fiftyTwoWeekHigh": float(overview_data.get("52WeekHigh", 0)) if overview_data.get("52WeekHigh") else "N/A",
                    "fiftyTwoWeekLow": float(overview_data.get("52WeekLow", 0)) if overview_data.get("52WeekLow") else "N/A",
                    "averageVolume": float(overview_data.get("AverageVolume", 0)) if overview_data.get("AverageVolume") else "N/A",
                    "longBusinessSummary": overview_data.get("Description", "No description available.")
                })
        except Exception as e:
            logger.warning(f"Could not fetch additional company info: {str(e)}")
        
        logger.info(f"Successfully retrieved data for {symbol}: {len(df)} rows")
        return df, company_info
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return None, f"Request error: {str(e)}"
    
    except ValueError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        return None, f"Error parsing API response: {str(e)}"
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return None, f"Unexpected error: {str(e)}"


def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    logger.debug(f"Calculating Bollinger Bands with window={window}, std={num_std}")
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    return rolling_mean, upper_band, lower_band


def format_number(num):
    """Format large numbers for display"""
    if num is None or num == "N/A":
        return "N/A"
    
    try:
        if num >= 1_000_000_000_000:
            return f"${num / 1_000_000_000_000:.2f}T"
        elif num >= 1_000_000_000:
            return f"${num / 1_000_000_000:.2f}B"
        elif num >= 1_000_000:
            return f"${num / 1_000_000:.2f}M"
        else:
            return f"${num:,.2f}"
    except Exception as e:
        logger.error(f"Error formatting number {num}: {str(e)}")
        return "N/A"


# Add delay between user actions to prevent API rate limiting
if "last_query_time" in st.session_state:
    time_since_last = time.time() - st.session_state.last_query_time
    if time_since_last < 15:
        logger.debug(f"Rate limiting local requests, waiting {15 - time_since_last:.2f}s")
        time.sleep(15 - time_since_last)
        
st.session_state.last_query_time = time.time()

# Main content
col1, col2 = st.columns([2, 1])

# Initialize placeholders for auto-refresh
chart_placeholder = st.empty()
info_placeholder = col2.empty()

# Function to update the data and charts
def update_data_and_charts():
    """Update the data and charts based on user selections"""
    logger.info("Updating data and charts")
    
    # Check if API key is set
    if ALPHA_VANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_API_KEY":
        st.error("Please enter your Alpha Vantage API key in the sidebar")
        return
    
    # Check if today is a trading day and adjust function
    current_date = dt.datetime.now()
    is_weekday = current_date.weekday() < 5  # Monday to Friday
    if is_weekday and function != "Intraday":
        av_function = "TIME_SERIES_INTRADAY"
        av_interval = "5min"  # Use a small interval for recent data
        logger.info("Switching to intraday data for current day")
    else:
        av_function = function_options[function]
        av_interval = interval_options[interval] if function == "Intraday" and interval else None
    
    av_output_size = output_size_options[output_size]
    logger.debug(f"Using Alpha Vantage params: function={av_function}, output_size={av_output_size}, interval={av_interval}")
    
    # Check for multiple stocks
    if "," in stock_input:
        with chart_placeholder.container():
            st.warning("Alpha Vantage API requires separate calls for each stock. Please select a single stock for detailed analysis.")
            st.info("For comparison charts, we'll fetch data for each stock separately.")
        
        # Handle multiple stocks for comparison
        tickers = stock_input.split(",")
        if len(tickers) > 5:
            st.error("Due to API limits, please select 5 or fewer stocks for comparison")
            return
        
        all_data = {}
        with st.spinner(f"Fetching data for {len(tickers)} stocks..."):
            for ticker in tickers:
                ticker = ticker.strip()
                logger.info(f"Fetching data for comparison: {ticker}")
                try:
                    df, _ = get_alpha_vantage_data(
                        symbol=ticker,
                        function=av_function,
                        output_size=av_output_size,
                        interval=av_interval
                    )
                    if df is not None:
                        all_data[ticker] = df
                        time.sleep(12)
                except Exception as e:
                    logger.error(f"Error fetching data for {ticker}: {str(e)}")
        
        # Create comparison chart
        with chart_placeholder.container():
            if all_data:
                st.header("Stock Price Comparison")
                
                fig = go.Figure()
                
                for ticker, df in all_data.items():
                    first_value = df['Close'].iloc[0]
                    normalized = df['Close'] / first_value * 100
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=normalized,
                            mode='lines',
                            name=ticker
                        )
                    )
                
                fig.update_layout(
                    title="Relative Performance (Base 100)",
                    xaxis_title="Date",
                    yaxis_title="Price (Normalized to 100)",
                    template="plotly_dark",
                    height=600,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not retrieve data for any of the selected stocks")
    
    else:
        # Single stock view
        with st.spinner(f"Fetching data for {stock_input}..."):
            try:
                data, info = get_alpha_vantage_data(
                    symbol=stock_input,
                    function=av_function,
                    output_size=av_output_size,
                    interval=av_interval
                )
                
                if data is None or data.empty:
                    logger.error(f"No data received for {stock_input}")
                    st.error(f"No data available for {stock_input} with the selected parameters")
                    st.info("This might be due to Alpha Vantage API limitations or an invalid ticker. Try another ticker or time period.")
                    return
                
                # Check if data is outdated
                if data.index[-1].date() < current_date.date():
                    st.warning(f"Latest data available is from {data.index[-1].date()}. The market might be closed today, or the API data is delayed.")
                
                logger.debug(f"Data shape: {data.shape}, Columns: {list(data.columns)}")
                logger.info(f"Successfully retrieved data for {stock_input}")
                    
            except Exception as e:
                logger.error(f"Error fetching data: {str(e)}", exc_info=True)
                st.error(f"Error fetching data: {str(e)}")
                st.info("Try selecting a different ticker symbol or time period.")
                return
        
        # Display stock info
        with info_placeholder.container():
            if info:
                company_name = info.get('shortName', stock_input)
                st.header(f"{company_name} ({stock_input})")
                logger.debug(f"Displaying info for {company_name}")
                
                try:
                    if 'currentPrice' in info and info['currentPrice']:
                        current_price = info['currentPrice']
                        price_change = info.get('change', 0)
                        price_change_pct = info.get('changePercent', '0%')
                        
                        if price_change >= 0:
                            emoji = "🔼"
                            color = "green"
                        else:
                            emoji = "🔽"
                            color = "red"
                            
                        st.markdown(f"""
                        <div style="font-size: 24px; font-weight: bold;">
                            ${current_price:.2f} 
                            <span style="color: {color};">
                                {emoji} {price_change:.2f} ({price_change_pct})
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        current_price = data['Close'].iloc[-1]
                        st.markdown(f"<div style='font-size: 24px; font-weight: bold;'>${current_price:.2f}</div>", unsafe_allow_html=True)
                except Exception as e:
                    logger.error(f"Error calculating price data: {str(e)}")
                    st.markdown("<div style='font-size: 24px; font-weight: bold;'>Price data unavailable</div>", unsafe_allow_html=True)
                
                st.subheader("Key Stats")
                logger.debug("Displaying key stats")
                
                stats_col1, stats_col2 = st.columns(2)
                
                try:
                    with stats_col1:
                        st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
                        st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
                        if info.get('marketCap'):
                            st.markdown(f"**Market Cap:** {format_number(info.get('marketCap'))}")
                        st.markdown(f"**52-Week Range:** ${info.get('fiftyTwoWeekLow', 'N/A')} - ${info.get('fiftyTwoWeekHigh', 'N/A')}")
                    
                    with stats_col2:
                        st.markdown(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
                        if info.get('dividendYield') not in (0, "N/A"):
                            st.markdown(f"**Dividend Yield:** {info.get('dividendYield', 0) * 100:.2f}%")
                        else:
                            st.markdown("**Dividend Yield:** N/A")
                        st.markdown(f"**Avg Volume:** {info.get('averageVolume', 'N/A'):,}")
                        st.markdown(f"**Beta:** {info.get('beta', 'N/A')}")
                except Exception as e:
                    logger.error(f"Error displaying stats: {str(e)}")
                    st.markdown("Some stats are unavailable.")
                
                with st.expander("Business Summary"):
                    st.write(info.get('longBusinessSummary', 'No summary available.'))
        
        # Create interactive chart
        try:
            with chart_placeholder.container():
                logger.debug(f"Creating {chart_type} chart")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.03, row_heights=[0.7, 0.3])
                
                if chart_type == "Candlestick":
                    fig.add_trace(
                        go.Candlestick(
                            x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'],
                            name="Price",
                            increasing_line_color='#26a69a', 
                            decreasing_line_color='#ef5350'
                        ),
                        row=1, col=1
                    )
                elif chart_type == "Line":
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['Close'],
                            mode='lines',
                            name="Price",
                            line=dict(color='#2962FF', width=2)
                        ),
                        row=1, col=1
                    )
                elif chart_type == "OHLC":
                    fig.add_trace(
                        go.Ohlc(
                            x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'],
                            name="Price",
                            increasing_line_color='#26a69a', 
                            decreasing_line_color='#ef5350'
                        ),
                        row=1, col=1
                    )
                
                if show_ma and len(data) > max(ma_periods if ma_periods else [0]):
                    logger.debug(f"Adding moving averages: {ma_periods}")
                    for ma_period in ma_periods:
                        if len(data) >= ma_period:
                            ma = data['Close'].rolling(window=ma_period).mean()
                            fig.add_trace(
                                go.Scatter(
                                    x=data.index,
                                    y=ma,
                                    name=f"{ma_period}-period MA",
                                    line=dict(width=1)
                                ),
                                row=1, col=1
                            )
                
                if show_bbands and len(data) >= 20:
                    logger.debug("Adding Bollinger Bands")
                    ma_20, upper_band, lower_band = calculate_bollinger_bands(data)
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=upper_band,
                            name="Upper BB",
                            line=dict(color='rgba(250,128,114,0.7)', width=1, dash='dash'),
                            showlegend=True
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=lower_band,
                            name="Lower BB",
                            line=dict(color='rgba(173,216,230,0.7)', width=1, dash='dash'),
                            showlegend=True
                        ),
                        row=1, col=1
                    )
                
                if show_volume and 'Volume' in data.columns:
                    logger.debug("Adding volume chart")
                    colors = ['rgba(38, 166, 154, 0.5)' if data['Close'].iloc[i] >= data['Open'].iloc[i] 
                             else 'rgba(239, 83, 80, 0.5)' 
                             for i in range(len(data))]
                    
                    fig.add_trace(
                        go.Bar(
                            x=data.index,
                            y=data['Volume'],
                            name="Volume",
                            marker_color=colors
                        ),
                        row=2, col=1
                    )
                
                fig.update_layout(
                    title=f"{company_name if info else stock_input} Stock Chart",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    template="plotly_dark",
                    xaxis_rangeslider_visible=False,
                    height=700,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                )
                
                fig.update_xaxes(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1d", step="day", stepmode="backward"),
                            dict(count=7, label="1w", step="day", stepmode="backward"),
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=3, label="3m", step="month", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    row=1, col=1
                )
                
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error creating chart: {str(e)}", exc_info=True)
            st.error(f"Error creating chart: {str(e)}")


# Initial data load or force refresh
logger.info("Performing initial data load")
if hasattr(st.session_state, 'force_refresh') and st.session_state.force_refresh:
    logger.info("Force refresh requested")
    st.session_state.force_refresh = False
    update_data_and_charts()
else:
    update_data_and_charts()

# Auto refresh functionality
if enable_auto_refresh:
    logger.info(f"Auto-refresh enabled with {refresh_interval}s interval")
    st.markdown(f"<div style='text-align: center; color: #4CAF50;'>Auto-refreshing every {refresh_interval} seconds</div>", unsafe_allow_html=True)
    
    if "counter" not in st.session_state:
        st.session_state.counter = 0
    
    placeholder = st.empty()
    
    try:
        while True:
            time.sleep(refresh_interval)
            st.session_state.counter += 1
            logger.info(f"Auto-refresh #{st.session_state.counter}")
            update_data_and_charts()
            placeholder.markdown(f"<div style='text-align: center; color: #4CAF50;'>Last updated: {dt.datetime.now().strftime('%H:%M:%S')} (refresh #{st.session_state.counter})</div>", unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error during auto-refresh: {str(e)}")
        st.error(f"Auto-refresh error: {str(e)}")

st.markdown("""
---
### About This App
This Streamlit application provides real-time stock market data using the Alpha Vantage API.
Built with Python, Streamlit, requests, and Plotly.

**Features:**
- Real-time stock quotes and charts
- Multiple chart types (Candlestick, Line, OHLC)
- Technical indicators (Moving Averages, Bollinger Bands)
- Multi-stock comparison
- Auto-refresh capabilities

**Get an API key:** [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
""")

# Show debug information in expander
with st.expander("Debug Information"):
    st.write(f"Log file: {log_filename}")
    st.write(f"Alpha Vantage API parameters: ticker={stock_input}, function={function_options[function]}")
    if function == "Intraday":
        st.write(f"Interval: {interval_options[interval]}")
    st.write(f"Output size: {output_size_options[output_size]}")
    st.write(f"Chart type: {chart_type}")
    st.write(f"Session retry config: max_retries={retry_strategy.total}, backoff_factor={retry_strategy.backoff_factor}")
    
    if st.button("View Recent Logs"):
        try:
            with open(log_filename, "r") as f:
                lines = f.readlines()
                recent_logs = lines[-50:] if len(lines) > 50 else lines
                st.code("".join(recent_logs), language="text")
        except Exception as e:
            st.error(f"Could not read log file: {str(e)}")

logger.info("Application UI fully rendered")