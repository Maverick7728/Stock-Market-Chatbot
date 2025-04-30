import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
import os
from datetime import datetime
import json
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = os.path.join(log_dir, f"stock_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO, # Changed to INFO for production, DEBUG is verbose
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
ALPHA_VANTAGE_API_KEY = "F9H662T7ZC52LER5" # Consider loading from env vars for security
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# --- Session State Initialization ---
if "selected_stocks" not in st.session_state:
    st.session_state.selected_stocks = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_query_time" not in st.session_state:
    st.session_state.last_query_time = 0
if "fetched_data" not in st.session_state:
    st.session_state.fetched_data = None # For single stock DataFrame
if "fetched_info" not in st.session_state:
    st.session_state.fetched_info = None # For single stock info dict
if "fetched_all_data" not in st.session_state:
    st.session_state.fetched_all_data = None # For multi-stock comparison
if "current_stock_context" not in st.session_state:
    st.session_state.current_stock_context = None # Store the symbol(s) data was fetched for
# --- End Session State Initialization ---

# Initialize different LLM models
def initialize_models():
    models = {}
    
    # Check for GROQ_API_KEY
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        groq_api_key = os.getenv("GROQ_API_KEY")
    
    try:
        # Llama 3.3 70B model
        models["Llama-3.3-70B"] = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=groq_api_key
        )
        
        # Gemma 2 9B model
        models["Gemma-2-9B"] = ChatGroq(
            model="gemma2-9b-it",
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=groq_api_key
        )
        
        # deepseek-r1-distill-llama-70b
        models["deepseek-r1-distill-llama-70b"] = ChatGroq(
            model="deepseek-r1-distill-llama-70b",
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=groq_api_key
        )
        
        return models
    except Exception as e:
        st.error(f"⚠️ Error initializing models: {str(e)}")
        return {}

# Get model response
def get_model_response(question, model):
    prompt = f"""You are a financial advisor specializing in stock market analysis.
USER QUESTION: {question}  
"""
    
    try:
        response = model.invoke(prompt)
        return {"content": response.content}
    except Exception as e:
        logger.error(f"Error invoking LLM: {str(e)}", exc_info=True)
        return {"content": f"Error communicating with the AI model: {str(e)}"}


# Helper function to summarize data for the LLM prompt
def summarize_data_for_llm(data=None, info=None, all_data=None, context=None):
    """Creates a concise summary of the fetched data for the LLM prompt."""
    summary = ""
    if info and data is not None and not data.empty: # Single stock context
        summary += f"Current context is for stock: {info.get('shortName', context)} ({context}).\n"
        summary += "Key Information:\n"
        summary += f"- Sector: {info.get('sector', 'N/A')}\n"
        summary += f"- Industry: {info.get('industry', 'N/A')}\n"
        latest_price = info.get('currentPrice', data['Close'].iloc[-1])
        summary += f"- Latest Price: ${latest_price:.2f}\n"
        if 'change' in info:
             summary += f"- Daily Change: {info.get('change', 0):.2f} ({info.get('changePercent', 'N/A')})\n"
        summary += f"- Market Cap: {format_number(info.get('marketCap', 'N/A'))}\n"
        summary += f"- P/E Ratio: {info.get('trailingPE', 'N/A')}\n"
        data_start = data.index.min().strftime('%Y-%m-%d')
        data_end = data.index.max().strftime('%Y-%m-%d')
        summary += f"- Displayed Data Range: {data_start} to {data_end}\n"
        # Add a snippet of recent price action
        recent_data = data.tail(5)
        summary += "- Recent Closing Prices:\n"
        for idx, row in recent_data.iterrows():
            summary += f"  - {idx.strftime('%Y-%m-%d')}: ${row['Close']:.2f}\n"

    elif all_data and context: # Multi-stock context
        summary += f"Current context is comparing stocks: {', '.join(context)}.\n"
        summary += "Latest Available Closing Prices:\n"
        for ticker, df in all_data.items():
            if not df.empty:
                latest_close = df['Close'].iloc[-1]
                latest_date = df.index.max().strftime('%Y-%m-%d')
                summary += f"- {ticker}: ${latest_close:.2f} (as of {latest_date})\n"
            else:
                 summary += f"- {ticker}: Data unavailable\n"
        # Can add date range if consistent, otherwise it might be complex
        # Example: Assume first df represents the range
        if context and all_data.get(context[0]) is not None:
             first_df = all_data[context[0]]
             if not first_df.empty:
                 data_start = first_df.index.min().strftime('%Y-%m-%d')
                 data_end = first_df.index.max().strftime('%Y-%m-%d')
                 summary += f"- Comparison Data Range Approx: {data_start} to {data_end}\n"

    return summary if summary else "No specific stock data is currently loaded."


# Get model response - MODIFIED to accept context
def get_model_response(question, model, fetched_data=None, fetched_info=None, fetched_all_data=None, current_context=None):
    """Generates response using the LLM, incorporating fetched data context."""

    # Create data summary for the prompt
    data_summary = summarize_data_for_llm(fetched_data, fetched_info, fetched_all_data, current_context)

    prompt = f"""You are a financial assistant embedded in a stock analysis app.
Your primary goal is to answer the user's questions about stocks and finance, focusing *specifically* on the data currently displayed or loaded in the app when relevant.

AVAILABLE DATA CONTEXT:
{data_summary}
---

USER QUESTION: {question}
---

YOUR TASK:
1.  Analyze the user's question.
2.  **If the question relates to the stock(s) mentioned in the 'AVAILABLE DATA CONTEXT', prioritize using the information provided in the context summary for your answer.** Refer to the latest price, trends, sector, etc., from the context.
3.  If the question is general financial knowledge or about stocks *not* in the current context, provide a helpful, general answer based on your knowledge.
4.  If you use the provided context, mention it subtly (e.g., "Based on the data loaded for {current_context}..." or "The current data shows...").
5.  Keep answers concise and factual.
6.  **Crucially, include the following disclaimer if providing any stock-specific information or analysis:** "Disclaimer: This information is for informational purposes only and does not constitute financial advice. Always conduct your own research or consult a qualified financial advisor before making investment decisions."
7.  If the question is clearly unrelated to finance or stocks, provide a polite and helpful answer, gently reminding the user of your specialization in financial topics.

Provide your response now.
"""
    logger.debug(f"Sending prompt to LLM (context summary length: {len(data_summary)} chars)")
    try:
        response = model.invoke(prompt)
        # Ensure disclaimer is present if needed (simple check)
        response_content = response.content
        if current_context and any(ticker in question.upper() for ticker in ([current_context] if isinstance(current_context, str) else current_context)):
             if "Disclaimer:" not in response_content:
                 response_content += "\n\nDisclaimer: This information is for informational purposes only and does not constitute financial advice. Always conduct your own research or consult a qualified financial advisor before making investment decisions."
                 logger.warning("Added missing disclaimer to LLM response.")

        return {"content": response_content}
    except Exception as e:
        logger.error(f"Error invoking LLM: {str(e)}", exc_info=True)
        return {"content": f"Error communicating with the AI model: {str(e)}"}


# Set page config
logger.debug("Setting up Streamlit page configuration")
st.set_page_config(
    page_title="Stock Market ChatBot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
/* ... [Your existing CSS remains unchanged] ... */
.main, .stApp {background-color: #0e1117; color: #fafafa;}
.stHeader {background-color: #262730;}
.stButton>button {background-color: #4CAF50; color: white; border-radius: 4px;}
.stSelectbox>div>div {background-color: #262730; color: white;}
.ticker-data {background-color: #262730; padding: 15px; border-radius: 5px; margin-bottom: 10px;}
.chat-container {background-color: #262730; padding: 15px; border-radius: 5px; margin-top: 20px;}
.stChatMessage {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# Initialize models
available_models = initialize_models()

# Title and description
st.title("📊 Stock Market ChatBot")
st.markdown("""
This app retrieves stock price data using Alpha Vantage, visualizes it, and provides AI-powered insights *based on the loaded data*.
Select stock(s), time period, and chart type. Then ask the AI assistant about the displayed information or general financial topics.
""")

# Updated sidebar section
with st.sidebar:
    st.header("Stock Selection & Chart Options")

    # AI Assistant model selection
    st.header("AI Assistant")
    selected_model = None # Initialize
    if available_models:
        selected_model_name = st.selectbox("Choose AI Model:", list(available_models.keys()))
        selected_model = available_models[selected_model_name] # Store the actual model object
        logger.info(f"AI Model selected: {selected_model_name}")
    else:
        st.warning("No AI models available. Check GROQ_API_KEY.")

    # Popular stock suggestions using session state
    popular_stocks = st.multiselect(
        "Select Stocks (Max 5 for comparison):",
        ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT",
         "DIS", "NFLX", "KO", "PFE", "INTC", "AMD", "BA", "MCD", "CSCO", "PG"],
        st.session_state.selected_stocks
    )

    # Limit selection for comparison
    if len(popular_stocks) > 5:
        st.warning("Please select a maximum of 5 stocks for comparison charts.")
        st.session_state.selected_stocks = popular_stocks[:5] # Keep only first 5
        # Need to rerun to reflect the change in multiselect widget state
        st.rerun()
    else:
         st.session_state.selected_stocks = popular_stocks

    if st.session_state.selected_stocks:
        if len(st.session_state.selected_stocks) == 1:
            stock_input = st.session_state.selected_stocks[0]
        else:
            stock_input = ",".join(st.session_state.selected_stocks)
        logger.info(f"Stock selection updated: {stock_input}")
    else:
        stock_input = ""

    # Time period selection (adapted for Alpha Vantage)
    output_size_options = {
        "Compact (Latest 100 data points)": "compact",
        "Full (All available data)": "full"
    }
    output_size = st.selectbox(
        "Select Data Size:", list(output_size_options.keys()), index=0
    )

    # Function/time series type selection
    function_options = {
        "Intraday": "TIME_SERIES_INTRADAY",
        "Daily": "TIME_SERIES_DAILY",
        "Weekly": "TIME_SERIES_WEEKLY",
        "Monthly": "TIME_SERIES_MONTHLY"
    }
    function = st.selectbox(
        "Select Time Series:", list(function_options.keys()), index=1 # Default to Daily
    )

    # Interval selection (only for intraday)
    interval_options = {
        "1 Minute": "1min", "5 Minutes": "5min", "15 Minutes": "15min",
        "30 Minutes": "30min", "60 Minutes": "60min"
    }
    interval = None
    if function == "Intraday":
        interval_key = st.selectbox(
            "Select Interval:", list(interval_options.keys()), index=2 # Default to 15min
        )
        interval = interval_options[interval_key] # Store the API value

    # Chart type
    chart_type = st.selectbox(
        "Select Chart Type:", ["Candlestick", "Line", "OHLC"], index=0
    )
    logger.info(f"Selected chart type: {chart_type}")

    # Technical indicators
    st.subheader("Technical Indicators")
    show_ma = st.checkbox("Moving Averages", value=True)
    ma_periods = []
    if show_ma:
        ma_periods = st.multiselect(
            "MA Periods:", [9, 20, 50, 100, 200], default=[20, 50]
        )
        logger.debug(f"Selected MA periods: {ma_periods}")


# Helper functions (get_alpha_vantage_data, calculate_bollinger_bands, format_number)
# remain largely the same as in your original code.
# Make sure get_alpha_vantage_data returns df, info or None, message

# --- [get_alpha_vantage_data function - unchanged from your code] ---
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

        if "Error Message" in data:
            logger.error(f"Alpha Vantage API error for {symbol}: {data['Error Message']}")
            return None, f"API Error for {symbol}: {data['Error Message']}"

        if "Note" in data:
            logger.warning(f"Alpha Vantage API note for {symbol}: {data['Note']}")
            # Avoid showing frequent user warnings for rate limits unless it prevents data fetch
            if "call frequency" not in data['Note'].lower():
                 st.warning(f"API Note for {symbol}: {data['Note']}")

        # Determine the correct time series key
        time_series_key = next((key for key in data if "Time Series" in key or "Weekly Time Series" in key or "Monthly Time Series" in key), None)

        if not time_series_key:
            logger.error(f"Could not find valid time series key in response for {symbol}")
            logger.debug(f"Response content sample for {symbol}: {json.dumps(data)[:500]}...")
            # Check for common issues like invalid symbol or API limit info
            if "Information" in data and "premium" in data["Information"].lower():
                 return None, f"API Limit Reached for {symbol}. Free tier has limitations. Please wait and try again."
            return None, f"Unexpected API response structure for {symbol}. Check ticker validity."

        # Convert to DataFrame
        time_series_data = data[time_series_key]
        df = pd.DataFrame.from_dict(time_series_data, orient="index")

        df.rename(columns={
            "1. open": "Open", "2. high": "High", "3. low": "Low",
            "4. close": "Close", "5. volume": "Volume"
        }, inplace=True)

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') # Use to_numeric for robustness

        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True) # Drop rows with NaNs in essential price columns

        # Get company info (handle potential errors gracefully)
        company_info = {"shortName": symbol} # Default name
        try:
            # Global Quote for price/change
            quote_params = {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": ALPHA_VANTAGE_API_KEY}
            quote_response = session.get(ALPHA_VANTAGE_BASE_URL, params=quote_params, timeout=10)
            quote_data = quote_response.json()
            if "Global Quote" in quote_data and quote_data["Global Quote"]:
                quote = quote_data["Global Quote"]
                company_info["currentPrice"] = float(quote.get("05. price", 0))
                company_info["change"] = float(quote.get("09. change", 0))
                company_info["changePercent"] = quote.get("10. change percent", "0%")

            # Company Overview for details
            overview_params = {"function": "OVERVIEW", "symbol": symbol, "apikey": ALPHA_VANTAGE_API_KEY}
            overview_response = session.get(ALPHA_VANTAGE_BASE_URL, params=overview_params, timeout=10)
            overview_data = overview_response.json()
            # Check if overview data is valid (AlphaVantage returns {} for invalid symbols sometimes)
            if "Symbol" in overview_data and overview_data["Symbol"] == symbol:
                company_info.update({
                    "shortName": overview_data.get("Name", symbol),
                    "sector": overview_data.get("Sector", "N/A"),
                    "industry": overview_data.get("Industry", "N/A"),
                    "marketCap": float(overview_data.get("MarketCapitalization", 0)) if overview_data.get("MarketCapitalization") != "None" else 0,
                    "beta": float(overview_data.get("Beta", 0)) if overview_data.get("Beta") != "None" else "N/A",
                    "trailingPE": float(overview_data.get("TrailingPE", 0)) if overview_data.get("TrailingPE") not in ["None", "-"] else "N/A",
                    "dividendYield": float(overview_data.get("DividendYield", 0)) if overview_data.get("DividendYield") not in ["None", "-"] else "N/A",
                    "fiftyTwoWeekHigh": float(overview_data.get("52WeekHigh", 0)) if overview_data.get("52WeekHigh") not in ["None", "-"] else "N/A",
                    "fiftyTwoWeekLow": float(overview_data.get("52WeekLow", 0)) if overview_data.get("52WeekLow") not in ["None", "-"] else "N/A",
                    "averageVolume": int(overview_data.get("Volume", 0)) if overview_data.get("Volume", '0').isdigit() else "N/A", # Use 'Volume' from overview if avg not there
                    "longBusinessSummary": overview_data.get("Description", "No description available.")
                })
            else:
                 logger.warning(f"Could not retrieve detailed company overview for {symbol}. Response: {overview_data}")


        except requests.exceptions.RequestException as e_info:
            logger.warning(f"Could not fetch additional company info for {symbol} due to request error: {str(e_info)}")
        except Exception as e_info:
            logger.warning(f"Could not parse additional company info for {symbol}: {str(e_info)}")

        logger.info(f"Successfully retrieved data for {symbol}: {len(df)} rows")
        return df, company_info

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error fetching {symbol}: {str(e)}")
        return None, f"Network error fetching {symbol}: {str(e)}"
    except json.JSONDecodeError as e:
         logger.error(f"JSON parsing error for {symbol}: {str(e)}. Response: {response.text[:200]}")
         return None, f"Error parsing API response for {symbol}."
    except Exception as e:
        logger.error(f"Unexpected error fetching {symbol}: {str(e)}", exc_info=True)
        return None, f"Unexpected error fetching {symbol}: {str(e)}"

# --- [calculate_bollinger_bands function - unchanged] ---
def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    logger.debug(f"Calculating Bollinger Bands with window={window}, std={num_std}")
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

# --- [format_number function - unchanged] ---
def format_number(num):
    """Format large numbers for display"""
    if num is None or num == "N/A" or not isinstance(num, (int, float)):
        return "N/A"
    try:
        num = float(num) # Ensure it's a float
        if abs(num) >= 1_000_000_000_000:
            return f"${num / 1_000_000_000_000:.2f}T"
        elif abs(num) >= 1_000_000_000:
            return f"${num / 1_000_000_000:.2f}B"
        elif abs(num) >= 1_000_000:
            return f"${num / 1_000_000:.2f}M"
        elif abs(num) >= 1000:
             return f"${num:,.2f}" # Add comma for thousands
        else:
            return f"${num:.2f}" # Format smaller numbers too
    except (ValueError, TypeError) as e:
        logger.error(f"Error formatting number {num}: {str(e)}")
        return "N/A"

# Add delay between user actions (Alpha Vantage free tier limit is strict)
API_CALL_DELAY = 15 # seconds delay between fetching data for different stocks
if time.time() - st.session_state.last_query_time < API_CALL_DELAY:
    wait_time = API_CALL_DELAY - (time.time() - st.session_state.last_query_time)
    logger.info(f"Rate limiting: Waiting {wait_time:.1f}s before next potential API call.")
    # No need to actually sleep here unless update_data_and_charts is called rapidly
    # The check within the multi-fetch loop is more critical.

# Layout for chart and company info
chart_col, info_col = st.columns([2, 1])

# Initialize placeholders
chart_placeholder = chart_col.empty()
info_placeholder = info_col.empty()

# Function to clear fetched data from session state
def clear_fetched_data_state():
    st.session_state.fetched_data = None
    st.session_state.fetched_info = None
    st.session_state.fetched_all_data = None
    st.session_state.current_stock_context = None
    logger.info("Cleared fetched data from session state.")

# Function to update the data and charts - MODIFIED to store data in session state
def update_data_and_charts():
    """Update the data and charts based on user selections and store data"""
    logger.info("Updating data and charts...")
    st.session_state.last_query_time = time.time() # Record time *before* potential API calls

    # Clear previous state before fetching new data
    clear_fetched_data_state()

    if not stock_input:
        with chart_placeholder.container():
            st.info("👆 Please select a stock from the sidebar to display graph")
        with info_placeholder.container():
            st.info("Select a stock to view detailed information")
        return # Exit if no stock is selected

    # Get data based on inputs
    av_function = function_options[function]
    av_output_size = output_size_options[output_size]
    av_interval_selected = interval # Already holds the API value like '15min' or None

    logger.debug(f"Using Alpha Vantage params: function={av_function}, output_size={av_output_size}, interval={av_interval_selected}")

    # Check for multiple stocks
    if "," in stock_input:
        tickers = [ticker.strip().upper() for ticker in stock_input.split(",")]
        logger.info(f"Multiple stocks selected for comparison: {tickers}")

        if len(tickers) > 5:
             # This check is also in the sidebar, but double-check here
            chart_placeholder.error("Error: Maximum 5 stocks allowed for comparison due to API limits.")
            info_placeholder.empty()
            return

        all_data = {}
        fetch_errors = []
        with chart_placeholder.container():
            st.info(f"Fetching comparison data for: {', '.join(tickers)}")
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, ticker in enumerate(tickers):
                 status_text.text(f"Fetching data for {ticker} ({i+1}/{len(tickers)})...")
                 logger.info(f"Fetching comparison data for {ticker}")
                 try:
                     # Use 'compact' for comparison charts unless 'Full' is explicitly needed and brief
                     # For simplicity, stick to user selection, but warn if 'Full' is chosen for many stocks
                     if av_output_size == "full" and len(tickers) > 2:
                          st.warning("Fetching 'Full' data for multiple stocks may be slow or hit API limits.")

                     df, _ = get_alpha_vantage_data( # Ignore company info for comparison chart
                         symbol=ticker,
                         function=av_function,
                         output_size=av_output_size,
                         interval=av_interval_selected
                     )
                     if df is not None and not df.empty:
                         all_data[ticker] = df
                         logger.info(f"Successfully fetched comparison data for {ticker}")
                     elif df is None:
                         # Error message already logged by get_alpha_vantage_data
                         fetch_errors.append(ticker)
                     else: # df is empty
                         logger.warning(f"No data returned for {ticker} with selected parameters.")
                         fetch_errors.append(f"{ticker} (No data)")


                     # Update progress
                     progress_bar.progress((i + 1) / len(tickers))

                     # Crucial delay to respect Alpha Vantage free tier limits (5 calls per minute)
                     if i < len(tickers) - 1: # Don't sleep after the last call
                         logger.debug(f"Waiting {API_CALL_DELAY}s before next API call...")
                         time.sleep(API_CALL_DELAY)
                         st.session_state.last_query_time = time.time() # Update last query time after sleep

                 except Exception as e:
                     logger.error(f"Error during multi-fetch loop for {ticker}: {str(e)}", exc_info=True)
                     fetch_errors.append(f"{ticker} (Fetch Error)")
                     progress_bar.progress((i + 1) / len(tickers)) # Still update progress

            status_text.text("Finished fetching data.")
            time.sleep(1) # Keep message visible briefly
            status_text.empty()
            progress_bar.empty()

        # Process results
        with chart_placeholder.container():
            if fetch_errors:
                st.warning(f"Could not retrieve data for: {', '.join(fetch_errors)}. Displaying available data.")

            if not all_data:
                st.error("Failed to retrieve data for any selected stock.")
                clear_fetched_data_state() # Ensure state is clear
                return

            # --- Store fetched multi-data in session state ---
            st.session_state.fetched_data = None
            st.session_state.fetched_info = None
            st.session_state.fetched_all_data = all_data
            st.session_state.current_stock_context = list(all_data.keys()) # Store tickers we got data for
            logger.info(f"Stored multi-stock data for {st.session_state.current_stock_context} in session state.")
            # --- End Store multi-data ---

            st.header("Stock Price Comparison")
            fig = go.Figure()
            base_value = 100 # Normalize to 100

            for ticker, df in all_data.items():
                if not df.empty and 'Close' in df.columns:
                    first_valid_close = df['Close'].dropna().iloc[0]
                    if first_valid_close > 0: # Avoid division by zero
                        normalized_close = (df['Close'] / first_valid_close) * base_value
                        fig.add_trace(go.Scatter(x=df.index, y=normalized_close, mode='lines', name=ticker))
                    else:
                        logger.warning(f"First closing price for {ticker} is zero, cannot normalize.")
                        # Optionally plot raw data or skip
                        # fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name=f"{ticker} (Raw)"))

            fig.update_layout(
                title=f"Relative Performance (Normalized to {base_value})",
                xaxis_title="Date", yaxis_title=f"Normalized Price (Base {base_value})",
                template="plotly_dark", height=600,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Clear info placeholder for multi-view
        info_placeholder.empty()

    else:
        # Single stock view
        single_ticker = stock_input.strip().upper()
        with chart_placeholder.container(): # Show spinner within the chart area
             with st.spinner(f"Fetching data for {single_ticker}..."):
                try:
                    data, info = get_alpha_vantage_data(
                        symbol=single_ticker,
                        function=av_function,
                        output_size=av_output_size,
                        interval=av_interval_selected
                    )

                    # Check for valid data
                    if data is None or data.empty:
                        error_message = info if isinstance(info, str) else f"No data available for {single_ticker} with the selected parameters."
                        logger.error(f"Data fetch failed or returned empty for {single_ticker}. Message: {error_message}")
                        st.error(error_message)
                        st.info("This could be due to an invalid ticker, API limits (free tier allows 5 calls/min, 100/day), or data unavailability for the chosen period/interval.")
                        clear_fetched_data_state()
                        info_placeholder.empty() # Clear info pane on error
                        return

                    logger.debug(f"Data shape for {single_ticker}: {data.shape}, Columns: {list(data.columns)}")
                    logger.info(f"Successfully retrieved data and info for {single_ticker}")

                    # --- Store fetched single-stock data in session state ---
                    st.session_state.fetched_data = data
                    st.session_state.fetched_info = info
                    st.session_state.fetched_all_data = None
                    st.session_state.current_stock_context = single_ticker
                    logger.info(f"Stored single-stock data for {single_ticker} in session state.")
                    # --- End Store single-data ---

                except Exception as e:
                    logger.error(f"Error in single stock fetch block for {single_ticker}: {str(e)}", exc_info=True)
                    st.error(f"An unexpected error occurred while fetching data for {single_ticker}: {str(e)}")
                    clear_fetched_data_state()
                    info_placeholder.empty()
                    return

        # Display stock info (using data from session state for consistency)
        current_info = st.session_state.fetched_info
        current_data = st.session_state.fetched_data
        current_ticker = st.session_state.current_stock_context

        with info_placeholder.container():
            if current_info and current_ticker:
                company_name = current_info.get('shortName', current_ticker)
                st.header(f"{company_name} ({current_ticker})")
                logger.debug(f"Displaying info for {company_name}")

                # Display price and change
                try:
                    current_price = current_info.get('currentPrice')
                    price_change = current_info.get('change')
                    price_change_pct = current_info.get('changePercent', 'N/A')

                    # Fallback to latest close if global quote failed
                    if current_price is None and current_data is not None and not current_data.empty:
                         current_price = current_data['Close'].iloc[-1]
                         # Cannot reliably calculate change without previous close from quote
                         price_change = None
                         logger.warning(f"Using latest close price for {current_ticker} as current price from quote was unavailable.")


                    if current_price is not None:
                        price_html = f"<div style='font-size: 24px; font-weight: bold;'>${current_price:.2f}"
                        if price_change is not None:
                            emoji = "🔼" if price_change >= 0 else "🔽"
                            color = "green" if price_change >= 0 else "red"
                            price_html += f"""
                                <span style="color: {color}; font-size: 18px; margin-left: 10px;">
                                    {emoji} {price_change:+.2f} ({price_change_pct})
                                </span>
                            """
                        price_html += "</div>"
                        st.markdown(price_html, unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='font-size: 20px; font-weight: bold;'>Price data unavailable</div>", unsafe_allow_html=True)

                except Exception as e:
                    logger.error(f"Error displaying price data for {current_ticker}: {str(e)}")
                    st.markdown("<div style='font-size: 20px; font-weight: bold;'>Price display error</div>", unsafe_allow_html=True)

                # Display key stats
                st.subheader("Key Stats")
                stats_col1, stats_col2 = st.columns(2)
                na_val = "N/A"
                with stats_col1:
                    st.markdown(f"**Sector:** {current_info.get('sector', na_val)}")
                    st.markdown(f"**Industry:** {current_info.get('industry', na_val)}")
                    st.markdown(f"**Market Cap:** {format_number(current_info.get('marketCap', na_val))}")
                    st.markdown(f"**52W High:** {format_number(current_info.get('fiftyTwoWeekHigh', na_val))}")


                with stats_col2:
                    pe = current_info.get('trailingPE', na_val)
                    st.markdown(f"**P/E Ratio:** {pe if pe != na_val else na_val}") # Basic formatting
                    div_yield = current_info.get('dividendYield', na_val)
                    st.markdown(f"**Div Yield:** {f'{div_yield*100:.2f}%' if isinstance(div_yield, float) and div_yield > 0 else na_val}")
                    st.markdown(f"**Avg Volume:** {current_info.get('averageVolume', na_val):,}" if isinstance(current_info.get('averageVolume'), int) else na_val)
                    st.markdown(f"**52W Low:** {format_number(current_info.get('fiftyTwoWeekLow', na_val))}")
                    # Beta is often less relevant for quick view, maybe move to expander?
                    # st.markdown(f"**Beta:** {current_info.get('beta', na_val)}")


                with st.expander("More Details"):
                     st.markdown(f"**Beta:** {current_info.get('beta', na_val)}")
                     st.markdown(f"**Description:** {current_info.get('longBusinessSummary', 'No summary available.')}")
            else:
                 # Should not happen if data fetch was successful, but as a fallback:
                 st.info("Company information not available.")


        # Create interactive chart (using data from session state)
        try:
            with chart_placeholder.container():
                if current_data is None or current_data.empty:
                     st.warning("Chart cannot be displayed as historical data is missing.")
                     return # Don't proceed with charting if data is bad

                logger.debug(f"Creating {chart_type} chart for {current_ticker}")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   vertical_spacing=0.03, row_heights=[0.7, 0.3])

                # Add main price chart trace
                chart_args = {'x': current_data.index, 'name': "Price"}
                if chart_type == "Candlestick":
                    chart_args.update({
                        'open': current_data['Open'], 'high': current_data['High'],
                        'low': current_data['Low'], 'close': current_data['Close'],
                        'increasing_line_color': '#26a69a', 'decreasing_line_color': '#ef5350'
                    })
                    fig.add_trace(go.Candlestick(**chart_args), row=1, col=1)
                elif chart_type == "Line":
                    chart_args.update({'y': current_data['Close'], 'mode': 'lines', 'line': dict(color='#2962FF', width=2)})
                    fig.add_trace(go.Scatter(**chart_args), row=1, col=1)
                elif chart_type == "OHLC":
                     chart_args.update({
                        'open': current_data['Open'], 'high': current_data['High'],
                        'low': current_data['Low'], 'close': current_data['Close'],
                        'increasing_line_color': '#26a69a', 'decreasing_line_color': '#ef5350'
                    })
                     fig.add_trace(go.Ohlc(**chart_args), row=1, col=1)


                # Add Moving Averages
                if show_ma and ma_periods:
                    logger.debug(f"Adding moving averages: {ma_periods}")
                    min_data_needed = max(ma_periods) if ma_periods else 0
                    if len(current_data) >= min_data_needed:
                        for ma_period in ma_periods:
                             if len(current_data) >= ma_period: # Check for each MA individually
                                ma = current_data['Close'].rolling(window=ma_period).mean()
                                fig.add_trace(
                                    go.Scatter(x=current_data.index, y=ma, name=f"{ma_period}-MA", line=dict(width=1)),
                                    row=1, col=1
                                )
                             else:
                                logger.warning(f"Not enough data points ({len(current_data)}) to calculate {ma_period}-period MA.")
                    elif len(current_data) > 0 :
                         logger.warning(f"Not enough data points ({len(current_data)}) for the longest MA period ({min_data_needed}). Skipping MAs.")


                # Add Volume chart
                if 'Volume' in current_data.columns and not current_data['Volume'].isnull().all():
                    logger.debug("Adding volume chart")
                    # Ensure 'Open' exists for coloring, else use default color
                    if 'Open' in current_data.columns:
                         colors = ['rgba(38, 166, 154, 0.5)' if row['Close'] >= row['Open'] else 'rgba(239, 83, 80, 0.5)'
                                  for _, row in current_data.iterrows()]
                    else:
                         colors = 'rgba(128, 128, 128, 0.5)' # Default grey if no Open data

                    fig.add_trace(go.Bar(x=current_data.index, y=current_data['Volume'], name="Volume", marker_color=colors), row=2, col=1)
                    fig.update_yaxes(title_text="Volume", row=2, col=1)
                else:
                     logger.debug("Volume data not available or all nulls, skipping volume chart.")


                # Update layout
                chart_title = f"{current_info.get('shortName', current_ticker)} ({current_ticker}) Stock Chart"
                fig.update_layout(
                    title=chart_title,
                    xaxis_title=None, # Hide x-axis title on top chart
                    yaxis_title="Price (USD)",
                    template="plotly_dark",
                    xaxis_rangeslider_visible=False, # Slider off by default for clarity
                    height=600,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified" # Better hover experience
                )
                fig.update_xaxes(showticklabels=True, row=1, col=1) # Ensure top x-axis ticks are visible if needed
                fig.update_xaxes(title_text="Date", row=2, col=1) # Label bottom x-axis

                # Add range selector buttons (optional, can clutter)
                # fig.update_xaxes(
                #     rangeselector=dict(
                #         buttons=list([
                #             dict(count=1, label="1m", step="month", stepmode="backward"),
                #             dict(count=6, label="6m", step="month", stepmode="backward"),
                #             dict(count=1, label="YTD", step="year", stepmode="todate"),
                #             dict(count=1, label="1y", step="year", stepmode="backward"),
                #             dict(step="all")
                #         ])
                #     ),
                #     row=1, col=1 # Attach selector to top chart's x-axis
                # )

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            logger.error(f"Error creating chart for {current_ticker}: {str(e)}", exc_info=True)
            st.error(f"Error creating chart: {str(e)}")


# Run the update function if stocks are selected
if stock_input:
    update_data_and_charts()
else:
    # Explicitly handle the case where no stock is selected (already handled in update func, but good practice)
    with chart_placeholder.container():
        st.info("👆 Please select a stock from the sidebar to display graph")
    with info_placeholder.container():
        st.info("Select a stock to view detailed information")
    clear_fetched_data_state() # Ensure state is clear if selection is removed


# --- AI Chatbot Interface - MODIFIED to use context ---
st.header("💬 Ask the AI Assistant")
st.markdown("Ask about the loaded stock data or general financial topics.")

# Chat container
chat_container = st.container()

with chat_container:
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input moved below the displayed messages
if prompt := st.chat_input("Ask about the data, stocks, or markets..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message immediately in the container
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    # Prepare and display assistant response
    with chat_container:
        with st.chat_message("assistant"):
            if selected_model: # Check if a model object is selected
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")

                # Retrieve context from session state
                data_context = st.session_state.get("fetched_data")
                info_context = st.session_state.get("fetched_info")
                all_data_context = st.session_state.get("fetched_all_data")
                stock_context = st.session_state.get("current_stock_context")

                logger.info(f"User prompt: {prompt}")
                logger.info(f"Context for LLM: Stock(s)={stock_context}, Has single data={data_context is not None}, Has multi data={all_data_context is not None}")

                with st.spinner("AI is analyzing..."):
                    response_data = get_model_response(
                        question=prompt,
                        model=selected_model, # Pass the actual model object
                        fetched_data=data_context,
                        fetched_info=info_context,
                        fetched_all_data=all_data_context,
                        current_context=stock_context
                    )

                message_placeholder.markdown(response_data["content"])
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_data["content"]})
                # Rerun slightly to ensure the message persists correctly if needed (often not necessary with containers)
                # st.rerun()
            else:
                st.warning("Please select an AI model from the sidebar to enable the chat.")
                # Optionally add this warning to the chat history as well
                st.session_state.messages.append({"role": "assistant", "content": "AI model not selected. Please choose one from the sidebar."})


# Add button to clear chat history and optionally reset stock selection
c1, c2 = st.columns([0.8, 0.2]) # Layout columns for button placement
with c2:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        logger.info("Chat history cleared by user.")
        st.rerun() # Rerun to clear the displayed chat


# App footer
st.markdown("---")
st.markdown("""
**Features:**
- Stock data via Alpha Vantage (Free tier limits apply: ~5 calls/min, ~100/day)
- Interactive Charts (Candlestick, Line, OHLC) & Moving Averages
- Single stock details & Multi-stock comparison (up to 5)
- Context-aware AI assistant for insights on loaded data & general finance
""")
st.caption(f"Log file: {log_filename}")