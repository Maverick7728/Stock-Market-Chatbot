# --- Import Libraries ---
import streamlit as st
import yfinance as yf # Keep for potential fallback or comparison
import datetime as dt
from datetime import datetime
import pandas as pd
import numpy as np
import time
import traceback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
import json
from langchain_groq import ChatGroq
import io # For BytesIO and saving Excel

# --- Configuration & Setup ---

# Load environment variables
load_dotenv()

# Set up logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = os.path.join(log_dir, f"stock_chat_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO, # INFO for production, DEBUG for development
    format="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("--- Application starting up ---")

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

# API Key Configurations
# Try getting AV key from environment, fallback to the hardcoded one if needed (not recommended for production)
ALPHA_VANTAGE_API_KEY_ENV = os.getenv("ALPHA_VANTAGE_API_KEY")
ALPHA_VANTAGE_API_KEY = ALPHA_VANTAGE_API_KEY_ENV if ALPHA_VANTAGE_API_KEY_ENV else "YOUR_ALPHA_VANTAGE_API_KEY" # Default if not in .env
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Check if API keys are present
if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_API_KEY":
    logger.warning("ALPHA_VANTAGE_API_KEY not found or using default. Alpha Vantage source may fail.")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not found. AI Assistant features will be disabled.")

# Data directory for storing Excel files
DATA_DIR = "stock_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    logger.info(f"Created data directory: {DATA_DIR}")

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Stock Analysis & AI Assistant",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS (Combined and refined)
st.markdown("""
<style>
/* General Styles */
.main, .stApp { background-color: #0e1117; color: #fafafa; }
.stHeader { background-color: #262730; }

/* Buttons */
.stButton>button {
    background-color: #4CAF50; /* Green */
    color: white;
    border-radius: 4px;
    padding: 10px 24px;
    border: none;
    transition: background-color 0.3s ease;
}
.stButton>button:hover { background-color: #45a049; }
.stDownloadButton>button {
    background-color: #1E88E5; /* Blue */
    color: white;
    border-radius: 4px;
    padding: 8px 18px;
    border: none;
    transition: background-color 0.3s ease;
    margin-left: 10px;
}
.stDownloadButton>button:hover { background-color: #1565C0; }

/* Inputs & Selects */
.stSelectbox>div>div { background-color: #262730; color: white; border: 1px solid #3d3d3d;}
.stTextInput>div>div>input { background-color: #262730; color: white; border: 1px solid #3d3d3d;}
.stTextArea>div>div>textarea { background-color: #262730; color: white; border: 1px solid #3d3d3d;}
.stDateInput>div>div>input { background-color: #262730; color: white; }
.stRadio>div { background-color: transparent; }

/* Cards & Containers */
.ticker-data { background-color: #262730; padding: 15px; border-radius: 5px; margin-bottom: 10px; border: 1px solid #3d3d3d;}
.stock-card {
    background-color: #2d2d2d;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    border: 1px solid #3d3d3d;
}
.info-card { /* Added from chatbot CSS */
    background-color: #2d2d2d;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
    border: 1px solid #3d3d3d;
}
.stExpander { border: 1px solid #3d3d3d; border-radius: 5px; }
.stExpander>div:first-child { background-color: #262730; }

/* Chat Interface */
.stChatMessage {
    background-color: #2d2d2d;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    color: #e0e0e0;
    border-left: 5px solid #4CAF50; /* Assistant accent */
    overflow-wrap: break-word;
    word-wrap: break-word;
}
div[data-testid="stChatMessage"][data-testid="chatAvatarIcon-user"] + div > div {
    background-color: #333940;
    border-left: 5px solid #1E88E5; /* Blue accent for user */
}
.stChatInput {
    background-color: #2d2d2d;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    color: #e0e0e0;
    border: 1px solid #3d3d3d;
}

/* Headings */
h1, h2, h3, h4, h5, h6 { color: #e0e0e0; }

/* Metrics */
div[data-testid="stMetric"] {
    background-color: #262730;
    padding: 12px;
    border-radius: 5px;
    text-align: center;
    border: 1px solid #3d3d3d;
}
div[data-testid="stMetric"] > label { font-weight: bold; color: #aaa; font-size: 0.9em; }
div[data-testid="stMetric"] > div:nth-of-type(2) { font-size: 1.4em; color: #fafafa; }
div[data-testid="stMetricDelta"] { display: flex; justify-content: center; align-items: center; padding-top: 5px; }
div[data-testid="stMetricDelta"] > div { font-size: 0.9em; margin-left: 5px; }

/* Progress bar */
.stProgress > div > div > div { background-color: #4CAF50; }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

# --- AI Model Initialization (from Chatbot) ---
def initialize_models():
    """Initialize and return available LLM models"""
    models = {}
    logger.info("Initializing LLM models...")

    if not GROQ_API_KEY:
        logger.warning("No GROQ API key found. LLM features will be disabled.")
        # st.sidebar.warning("Groq API Key missing. AI Assistant disabled.", icon="🤖") # Warning shown elsewhere
        return {}

    # Define models to try initializing (Using models from chatbot example)
    model_configs = {
        "Llama-3.1-70B": "llama-3.1-70b-versatile", # Renamed for consistency
        "Gemma-2-9B": "gemma2-9b-it",
        # Add Llama 3.1 8B as a faster option if needed
        "Llama-3.1-8B": "llama-3.1-8b-instant",
    }

    initialized_models = {}
    for name, model_id in model_configs.items():
        try:
            initialized_models[name] = ChatGroq(
                model=model_id,
                temperature=0.7,
                max_tokens=None, # Let model decide
                timeout=None,
                max_retries=2,
                api_key=GROQ_API_KEY
            )
            logger.info(f"Successfully initialized model: {name} ({model_id})")
        except Exception as e:
            logger.error(f"Failed to initialize model {name} ({model_id}): {str(e)}")
            # st.sidebar.warning(f"Failed to load model: {name}", icon="⚠️") # Moved warning logic

    if not initialized_models:
        logger.error("Failed to initialize any LLM models.")
        # st.sidebar.error("Could not load any AI models!", icon="🔥") # Moved error logic
    else:
        logger.info(f"Successfully initialized {len(initialized_models)} models: {list(initialized_models.keys())}")

    return initialized_models

# --- Data Fetching Functions ---
# Enhanced get_alpha_vantage_data combining robustness from script 1 and AV params from script 2
def get_alpha_vantage_data(symbol, function, output_size="compact", interval=None):
    """Fetch stock data from Alpha Vantage API, including company info. Returns (df, info_dict_or_error_string)"""
    if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_API_KEY":
        logger.error(f"Cannot fetch Alpha Vantage data for {symbol}: API key not configured.")
        return None, "Alpha Vantage API key not configured or invalid."

    logger.info(f"Fetching Alpha Vantage data for {symbol} (func: {function}, size: {output_size}, interval: {interval})")

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
        response = session.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        logger.debug(f"Raw AV response keys for {symbol} ({function}): {list(data.keys())}")

        # --- Check for common issues ---
        if not data:
             return None, f"Received empty response from Alpha Vantage for {symbol} ({function})."
        if "Error Message" in data:
            return None, f"API error for {symbol}: {data['Error Message']}"
        if "Note" in data:
            note_msg = f"API limit note for {symbol}: {data['Note']}"
            logger.warning(note_msg)
            st.toast(f"⚠️ {note_msg}", icon="⚠️") # Use toast for less intrusive warnings

        # --- Determine the correct key for time series data ---
        time_series_key = ""
        possible_keys = []
        if function == "TIME_SERIES_INTRADAY":
            possible_keys.append(f"Time Series ({interval})")
        elif function == "TIME_SERIES_DAILY":
            possible_keys.append("Time Series (Daily)")
        elif function == "TIME_SERIES_WEEKLY":
            possible_keys.append("Weekly Time Series")
        elif function == "TIME_SERIES_MONTHLY":
            possible_keys.append("Monthly Time Series")

        # Attempt fallback to daily if intraday key not found (common issue)
        if function == "TIME_SERIES_INTRADAY" and possible_keys[0] not in data:
             fallback_key = "Time Series (Daily)"
             if fallback_key in data:
                 logger.warning(f"Intraday key '{possible_keys[0]}' not found for {symbol}, falling back to '{fallback_key}'.")
                 possible_keys.append(fallback_key)
             else: # Try refetching daily explicitly if fallback key also missing
                logger.warning(f"Intraday key '{possible_keys[0]}' and fallback '{fallback_key}' not found for {symbol}. Attempting explicit daily fetch.")
                daily_params = params.copy()
                daily_params["function"] = "TIME_SERIES_DAILY"
                if "interval" in daily_params: del daily_params["interval"]
                try:
                    daily_response = session.get(ALPHA_VANTAGE_BASE_URL, params=daily_params, timeout=15)
                    daily_response.raise_for_status()
                    data = daily_response.json() # Overwrite original data with daily response
                    function = "TIME_SERIES_DAILY" # Update function variable
                    possible_keys = ["Time Series (Daily)"] # Reset possible keys
                    logger.info(f"Successfully fetched daily data for {symbol} after intraday failure.")
                except Exception as daily_err:
                     logger.error(f"Failed to fetch fallback daily data for {symbol}: {daily_err}")
                     # Continue with original data/error check below

        # Find the first valid key
        for key in possible_keys:
            if key in data:
                time_series_key = key
                break

        if not time_series_key:
            return None, f"Could not find expected time series data key in response for {symbol}. Found keys: {list(data.keys())}"

        # --- Convert time series data to DataFrame ---
        time_series_data = data.get(time_series_key)
        df = pd.DataFrame() # Initialize empty DF
        if time_series_data:
            try:
                df = pd.DataFrame.from_dict(time_series_data, orient="index")
                df.rename(columns={
                    "1. open": "Open", "2. high": "High", "3. low": "Low",
                    "4. close": "Close", "5. volume": "Volume",
                    "6. adjusted close": "Adj Close",
                    "7. dividend amount": "Dividend Amount",
                    "8. split coefficient": "Split Coefficient"
                }, inplace=True)

                numeric_cols = ["Open", "High", "Low", "Close", "Volume", "Adj Close", "Dividend Amount", "Split Coefficient"]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
                logger.info(f"Successfully parsed time series data for {symbol}. Rows: {len(df)}")
            except Exception as parse_err:
                logger.error(f"Error parsing time series data for {symbol}: {parse_err}", exc_info=True)
                # Don't fail entirely, try to return info if possible
                df = pd.DataFrame() # Ensure df is empty
        else:
             logger.warning(f"Time series data under key '{time_series_key}' is empty for {symbol}.")


        # --- Fetch company overview and quote for additional info ---
        company_info = {"symbol": symbol, "source": "Alpha Vantage"}
        # Fetch Global Quote
        try:
            logger.debug(f"Fetching Global Quote for {symbol} (Alpha Vantage)")
            quote_params = {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": ALPHA_VANTAGE_API_KEY}
            quote_response = session.get(ALPHA_VANTAGE_BASE_URL, params=quote_params, timeout=10)
            quote_response.raise_for_status()
            quote_data = quote_response.json().get("Global Quote", {}) # Safely get dict
            if quote_data:
                company_info["currentPrice"] = pd.to_numeric(quote_data.get("05. price"), errors='coerce')
                company_info["change"] = pd.to_numeric(quote_data.get("09. change"), errors='coerce')
                company_info["changePercent"] = quote_data.get("10. change percent", "N/A") # Keep as string
                company_info["volume"] = pd.to_numeric(quote_data.get("06. volume"), errors='coerce')
                company_info["latestTradingDay"] = quote_data.get("07. latest trading day")
            elif "Note" in quote_response.json(): # Check original response for notes
                 logger.warning(f"API limit note (Global Quote) for {symbol}: {quote_response.json()['Note']}")
            elif "Error Message" in quote_response.json():
                 logger.warning(f"API error (Global Quote) for {symbol}: {quote_response.json()['Error Message']}")
        except Exception as e:
             logger.warning(f"Failed to fetch Alpha Vantage Global Quote for {symbol}: {e}")

        # Fetch Company Overview
        try:
            logger.debug(f"Fetching Overview for {symbol} (Alpha Vantage)")
            overview_params = {"function": "OVERVIEW", "symbol": symbol, "apikey": ALPHA_VANTAGE_API_KEY}
            overview_response = session.get(ALPHA_VANTAGE_BASE_URL, params=overview_params, timeout=15)
            overview_response.raise_for_status()
            overview_data = overview_response.json()
            if overview_data and overview_data.get("Symbol") == symbol:
                company_info.update({
                    "shortName": overview_data.get("Name", symbol),
                    "longBusinessSummary": overview_data.get("Description", "No description available."),
                    "sector": overview_data.get("Sector", "N/A"),
                    "industry": overview_data.get("Industry", "N/A"),
                    "marketCap": pd.to_numeric(overview_data.get("MarketCapitalization"), errors='coerce'),
                    # Use correct keys from AV Overview
                    "trailingPE": pd.to_numeric(overview_data.get("PERatio"), errors='coerce'),
                    "forwardPE": pd.to_numeric(overview_data.get("ForwardPE"), errors='coerce'),
                    "dividendYield": pd.to_numeric(overview_data.get("DividendYield"), errors='coerce'),
                    "fiftyTwoWeekHigh": pd.to_numeric(overview_data.get("52WeekHigh"), errors='coerce'),
                    "fiftyTwoWeekLow": pd.to_numeric(overview_data.get("52WeekLow"), errors='coerce'),
                    "beta": pd.to_numeric(overview_data.get("Beta"), errors='coerce'),
                    "sharesOutstanding": pd.to_numeric(overview_data.get("SharesOutstanding"), errors='coerce'),
                    # Calculate Avg Vol if needed, AV Overview doesn't have it directly
                    # "averageVolume": pd.to_numeric(overview_data.get("AverageVolume", 0)) # Example if key existed
                })
            elif "Note" in overview_data:
                logger.warning(f"API limit note (Overview) for {symbol}: {overview_data['Note']}")
            elif "Error Message" in overview_data:
                logger.warning(f"API error (Overview) for {symbol}: {overview_data['Error Message']}")
        except Exception as e:
            logger.warning(f"Failed to fetch Alpha Vantage Overview for {symbol}: {e}")

        # Ensure essential fields
        if "shortName" not in company_info: company_info["shortName"] = symbol

        return df, company_info # Return DataFrame (possibly empty) and info dict

    except requests.exceptions.RequestException as e:
        return None, f"Network error fetching Alpha Vantage data for {symbol}: {str(e)}"
    except json.JSONDecodeError as e:
        try: response_text = response.text
        except: response_text = "(Could not retrieve response text)"
        logger.error(f"JSON parsing error for {symbol}: {e}. Response: {response_text[:500]}...")
        return None, f"Error parsing JSON response from Alpha Vantage for {symbol}."
    except Exception as e:
        logger.error(f"Unexpected error fetching Alpha Vantage data for {symbol}: {str(e)}", exc_info=True)
        return None, f"Unexpected error fetching Alpha Vantage data for {symbol}: {str(e)}"

# Keep other helper functions from script 1
def get_yahoo_finance_data(symbol, period=None, start=None, end=None):
    """Fetch stock data from Yahoo Finance. Returns (df, info_dict_or_error_string)"""
    logger.info(f"Fetching Yahoo Finance data for {symbol} (Period: {period}, Start: {start}, End: {end})")
    try:
        stock = yf.Ticker(symbol)
        hist = None
        if period == "Custom" and start and end:
            # yfinance end date is exclusive, add one day to include it
            end_date_yf = end + pd.Timedelta(days=1)
            hist = stock.history(start=start, end=end_date_yf, timeout=20)
        elif period:
            hist = stock.history(period=period, timeout=20)
        else: # Default if no period/dates specified (e.g., 1 year)
             hist = stock.history(period="1y", timeout=20)

        if hist is not None and not hist.empty:
            df = hist
            info_dict = {"symbol": symbol, "shortName": symbol, "source": "Yahoo Finance"} # Basic info
            try:
                info_data = stock.info
                # Add source marker and symbol if missing (shouldn't be but safe)
                info_data["source"] = "Yahoo Finance"
                info_data["symbol"] = symbol
                # Standardize keys slightly if possible (e.g., marketCap)
                info_dict.update(info_data)
                logger.debug(f"Successfully fetched yfinance info for {symbol}")
            except Exception as info_err:
                logger.warning(f"Could not fetch yfinance info for {symbol}: {info_err}")
                info_dict["error_info"] = f"Could not fetch details: {info_err}"
            return df, info_dict
        else:
            return None, f"No data returned from Yahoo Finance for {symbol} ({period or f'{start} to {end}'})."

    except Exception as e:
        logger.error(f"Error fetching Yahoo Finance data for {symbol}: {e}", exc_info=True)
        return None, f"Error fetching Yahoo Finance data for {symbol}: {e}"

def get_data_from_excel_files(symbols=None):
    """Load latest stock data from saved Excel files for given symbols or all unique symbols."""
    # Re-using the robust function from script 1
    logger.info(f"Attempting to load data from Excel files. Symbols requested: {symbols}")
    excel_data = {} # Structure: {symbol: {"data": df, "info": dict, "file": str, "timestamp": str}}
    summary_parts = []

    try:
        if not os.path.exists(DATA_DIR):
             logger.warning(f"Data directory '{DATA_DIR}' does not exist.")
             return {"text": f"Saved data directory '{DATA_DIR}' not found.", "data": {}}

        files = os.listdir(DATA_DIR)
        excel_files = sorted(
            [f for f in files if f.endswith('.xlsx') and '_' in f and not f.startswith('~')],
            reverse=True
        )

        if not excel_files:
            return {"text": "No saved stock data files found.", "data": {}}

        processed_symbols = set()
        files_to_load = []

        if symbols:
            symbols_to_find = set(s.upper() for s in symbols)
            logger.debug(f"Searching for latest files for: {symbols_to_find}")
            for f in excel_files:
                try:
                    symbol_from_file = f.split('_')[0].upper()
                    if symbol_from_file in symbols_to_find and symbol_from_file not in processed_symbols:
                        files_to_load.append(f)
                        processed_symbols.add(symbol_from_file)
                        if len(processed_symbols) == len(symbols_to_find): break
                except IndexError: logger.warning(f"Could not parse symbol from filename: {f}. Skipping.")
            missing_symbols = symbols_to_find - processed_symbols
            if missing_symbols: st.toast(f"⚠️ No saved data found for: {', '.join(missing_symbols)}", icon="⚠️")
        else:
            logger.debug("No specific symbols requested, finding latest file for all unique symbols.")
            for f in excel_files:
                try:
                    symbol_from_file = f.split('_')[0].upper()
                    if symbol_from_file not in processed_symbols:
                        files_to_load.append(f)
                        processed_symbols.add(symbol_from_file)
                except IndexError: logger.warning(f"Could not parse symbol from filename: {f}. Skipping.")
            MAX_LOAD_ALL = 10
            if len(files_to_load) > MAX_LOAD_ALL:
                logger.warning(f"Found {len(files_to_load)} unique symbols. Loading the {MAX_LOAD_ALL} most recent.")
                files_to_load = files_to_load[:MAX_LOAD_ALL]

        if not files_to_load:
             load_msg = f"No saved data files found for symbols: {', '.join(symbols)}." if symbols else "No relevant saved data files found."
             return {"text": load_msg, "data": {}}

        logger.info(f"Loading data from files: {files_to_load}")
        st.progress(0.0)
        for i, file in enumerate(files_to_load):
            file_path = os.path.join(DATA_DIR, file)
            try:
                symbol = file.split('_')[0].upper()
                timestamp_str = file.split('_')[1].split('.')[0] if len(file.split('_')) > 1 else "UnknownTime"
                df = pd.read_excel(file_path, sheet_name='PriceData', index_col=0, engine='openpyxl')
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)

                metadata = {"symbol": symbol, "source": "Saved Excel", "shortName": symbol}
                try:
                    meta_df = pd.read_excel(file_path, sheet_name='Metadata', engine='openpyxl')
                    if 'Attribute' in meta_df.columns and 'Value' in meta_df.columns:
                        loaded_meta = {row['Attribute']: row['Value'] for _, row in meta_df.iterrows() if pd.notna(row['Attribute'])}
                        metadata.update(loaded_meta)
                        if 'shortName' not in metadata or pd.isna(metadata.get('shortName')):
                            metadata['shortName'] = symbol
                except Exception as sheet_error:
                    logger.warning(f"Could not read 'Metadata' sheet from {file}: {sheet_error}.")

                excel_data[symbol] = {"data": df, "info": metadata, "file": file, "timestamp": timestamp_str}

                company_name = metadata.get("shortName", symbol)
                last_price_str = "N/A"; change_str = ""
                if not df.empty and 'Close' in df.columns and pd.notna(df['Close'].iloc[-1]):
                    last_price = df['Close'].iloc[-1]; last_price_str = f"${last_price:.2f}"
                    if len(df['Close'].dropna()) >= 2:
                        prev_close = df['Close'].dropna().iloc[-2]
                        if pd.notna(prev_close) and prev_close != 0:
                            price_change = last_price - prev_close; change_percent = (price_change / prev_close) * 100
                            change_direction = "🔼" if price_change >= 0 else "🔽"
                            change_str = f" {change_direction} {price_change:+.2f} ({change_percent:+.2f}%)"
                        else: change_str = " (Change N/A)"
                    else: change_str = " (Change N/A)"
                try: dt_obj = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S'); formatted_time = dt_obj.strftime('%Y-%m-%d %H:%M')
                except ValueError: formatted_time = timestamp_str
                summary_parts.append(f"- **{company_name} ({symbol})**: Last {last_price_str}{change_str}. (Saved: {formatted_time})")

            except Exception as read_error:
                logger.error(f"Error reading Excel file {file}: {read_error}", exc_info=True)
                st.error(f"Failed to load data from {file}: {read_error}")
            st.progress((i + 1) / len(files_to_load))

        if not excel_data:
            return {"text": "Failed to load data from found Excel files.", "data": {}}

        final_summary = "Loaded data from saved files:\n" + "\n".join(sorted(summary_parts))
        logger.info(f"Successfully loaded data for symbols: {list(excel_data.keys())} from {len(files_to_load)} files.")
        return {"text": final_summary, "data": excel_data}

    except Exception as e:
        logger.error(f"Error accessing or listing files in {DATA_DIR}: {e}", exc_info=True)
        st.error(f"Could not access saved data directory: {e}")
        return {"text": "Error accessing saved data directory.", "data": {}}

def save_data_to_excel(symbol, df, info=None):
    """Save stock data to Excel (in memory) and return filename, bytes content."""
    # Re-using function from script 1
    if df is None or df.empty:
        logger.warning(f"No data provided to save for {symbol}")
        return None, None

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_{timestamp}.xlsx"
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='PriceData', index=True)
            if info and isinstance(info, dict):
                serializable_info = {}
                for k, v in info.items():
                    if isinstance(v, (str, int, float, bool, type(None))): serializable_info[k] = v
                    elif isinstance(v, (np.int64, np.float64)): serializable_info[k] = v.item()
                    elif isinstance(v, (dt.date, dt.datetime)): serializable_info[k] = v.isoformat()
                if serializable_info:
                     meta_df = pd.DataFrame(list(serializable_info.items()), columns=['Attribute', 'Value'])
                     meta_df.to_excel(writer, sheet_name='Metadata', index=False)
        excel_content = output.getvalue()
        # Optionally save locally as well
        # try:
        #     with open(os.path.join(DATA_DIR, filename), 'wb') as f: f.write(excel_content)
        #     logger.info(f"Saved data for {symbol} locally to {filename}")
        # except Exception as write_err: logger.error(f"Error writing local file {filename}: {write_err}")
        return filename, excel_content
    except Exception as e:
        logger.error(f"Error creating Excel data in memory for {symbol}: {str(e)}", exc_info=True)
        st.error(f"Failed to prepare Excel data for {symbol}: {e}")
        return None, None

# --- Charting Function (from script 1) ---
def generate_stock_chart(data, company_info=None, chart_type="Candlestick", show_ma=True, ma_periods=[20, 50], show_volume=True, show_bbands=False):
    """Generate an interactive stock chart using Plotly"""
    if data is None or data.empty: return None
    symbol = company_info.get('shortName', 'Stock') if company_info else 'Stock'
    logger.info(f"Generating '{chart_type}' chart for {symbol}...")
    try:
        rows = 2 if show_volume and 'Volume' in data.columns else 1
        row_heights = [0.7, 0.3] if rows == 2 else [1.0]
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                          vertical_spacing=0.03, row_heights=row_heights,
                          subplot_titles=("Price Chart", "Volume" if rows == 2 else None))
        price_chart_row = 1; price_chart_col = 1
        required_ohlc = ['Open', 'High', 'Low', 'Close']; has_ohlc = all(col in data.columns for col in required_ohlc)
        has_close = 'Close' in data.columns
        plot_type = None
        if chart_type in ["Candlestick", "OHLC"] and has_ohlc: plot_type = chart_type
        elif has_close: plot_type = "Line"
        else: st.error(f"Cannot plot for {symbol}: Missing 'Close' column."); return None

        if plot_type == "Candlestick": fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Price", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=price_chart_row, col=price_chart_col)
        elif plot_type == "Line": fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name="Price", line=dict(color='#2962FF', width=2)), row=price_chart_row, col=price_chart_col)
        elif plot_type == "OHLC": fig.add_trace(go.Ohlc(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Price", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=price_chart_row, col=price_chart_col)

        if show_ma and has_close and ma_periods:
            ma_colors = ['#FFD700', '#ADFF2F', '#FFA07A', '#00CED1', '#DA70D6']; color_idx = 0; valid_ma_periods = []
            for period in sorted(ma_periods):
                if isinstance(period, int) and period > 0:
                    if len(data) >= period:
                        ma = data['Close'].rolling(window=period, min_periods=1).mean(); color = ma_colors[color_idx % len(ma_colors)]
                        fig.add_trace(go.Scatter(x=data.index, y=ma, name=f"MA {period}", mode='lines', line=dict(width=1.5, color=color), opacity=0.8), row=price_chart_row, col=price_chart_col)
                        color_idx += 1; valid_ma_periods.append(period)
                    else: logger.warning(f"Not enough data ({len(data)}) for MA {period} for {symbol}.")
                else: logger.warning(f"Invalid MA period '{period}' skipped for {symbol}.")
            ma_periods = valid_ma_periods

        bb_period = 20; bb_std_dev = 2
        if show_bbands and has_close and len(data) >= bb_period:
            ma_bb = data['Close'].rolling(window=bb_period, min_periods=1).mean(); std_bb = data['Close'].rolling(window=bb_period, min_periods=1).std()
            upper_band = ma_bb + (std_bb * bb_std_dev); lower_band = ma_bb - (std_bb * bb_std_dev)
            fig.add_trace(go.Scatter(x=data.index, y=upper_band, name=f"Upper BB ({bb_period},{bb_std_dev})", mode='lines', line=dict(color='rgba(173, 216, 230, 0.5)', width=1, dash='dash'), fill=None), row=price_chart_row, col=price_chart_col)
            fig.add_trace(go.Scatter(x=data.index, y=lower_band, name=f"Lower BB ({bb_period},{bb_std_dev})", mode='lines', line=dict(color='rgba(173, 216, 230, 0.5)', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.1)'), row=price_chart_row, col=price_chart_col)
            if not show_ma or bb_period not in ma_periods:
                 fig.add_trace(go.Scatter(x=data.index, y=ma_bb, name=f"MA {bb_period} (BB)", mode='lines', line=dict(color='rgba(250, 128, 114, 0.7)', width=1)), row=price_chart_row, col=price_chart_col)

        if rows == 2:
             volume_chart_row = 2; volume_chart_col = 1
             if 'Volume' in data.columns:
                 if has_ohlc: colors = ['#26a69a' if data['Close'].iloc[i] >= data['Open'].iloc[i] else '#ef5350' for i in range(len(data))]
                 else: colors = 'rgba(128, 128, 128, 0.7)'
                 fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name="Volume", marker_color=colors, opacity=0.7), row=volume_chart_row, col=volume_chart_col)
                 fig.update_yaxes(title_text="Volume", row=volume_chart_row, col=volume_chart_col, showgrid=False)
             else: logger.warning(f"Volume display requested but 'Volume' column missing for {symbol}.")

        fig.update_layout(yaxis_title="Price (USD)", template="plotly_dark", height=650, xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=50, r=50, t=50, b=50), hovermode="x unified")
        if rows == 2: fig.update_xaxes(rangeslider_visible=False, row=volume_chart_row, col=volume_chart_col)
        fig.update_xaxes(rangeselector=dict(buttons=list([dict(count=1, label="1m", step="month", stepmode="backward"), dict(count=3, label="3m", step="month", stepmode="backward"), dict(count=6, label="6m", step="month", stepmode="backward"), dict(count=1, label="YTD", step="year", stepmode="todate"), dict(count=1, label="1y", step="year", stepmode="backward"), dict(count=5, label="5y", step="year", stepmode="backward"), dict(step="all", label="Max")]), bgcolor="#333", activecolor="#4CAF50", font=dict(color="#fafafa")), type="date", row=price_chart_row, col=price_chart_col)
        logger.info(f"Successfully generated chart for {symbol}.")
        return fig
    except Exception as e:
        logger.error(f"Error generating chart for {symbol}: {str(e)}", exc_info=True)
        st.error(f"Could not generate chart for {symbol}: {e}")
        return None

# Keep market overview chart function
def generate_market_overview_chart(period="3mo"):
    """Generate a comparison chart of major market indices using Yahoo Finance"""
    logger.info(f"Generating market overview chart for period: {period}")
    try:
        indices = {"S&P 500": "^GSPC", "Dow Jones": "^DJI", "NASDAQ": "^IXIC", "Russell 2000": "^RUT", "VIX": "^VIX"}
        vix_symbol = "^VIX"; index_symbols = [s for s in indices.values() if s != vix_symbol]
        tickers_str = " ".join(indices.values())
        logger.debug(f"Downloading market data for: {tickers_str} (period: {period})")
        hist_data = yf.download(tickers_str, period=period, progress=False, timeout=30)

        if hist_data.empty or 'Close' not in hist_data: return None
        perf_data = pd.DataFrame(); vix_data = None
        for name, symbol in indices.items():
            if symbol == vix_symbol: continue
            close_col = ('Close', symbol)
            if close_col in hist_data.columns:
                index_hist = hist_data[close_col].dropna()
                if not index_hist.empty:
                    first_valid_price = index_hist.iloc[0]
                    if first_valid_price and first_valid_price != 0: perf_data[name] = ((index_hist / first_valid_price) - 1) * 100
                    else: logger.warning(f"First price for {name} ({symbol}) is zero or NaN.")
                else: logger.warning(f"No closing data for index {name} ({symbol}).")
            else: logger.warning(f"Column '{close_col}' not found for index {name} ({symbol}).")
        vix_close_col = ('Close', vix_symbol)
        if vix_close_col in hist_data.columns: vix_data = hist_data[vix_close_col].dropna(); vix_data = None if vix_data.empty else vix_data
        else: logger.warning(f"Column '{vix_close_col}' not found for VIX ({vix_symbol}).")
        if perf_data.empty and vix_data is None: return None

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if not perf_data.empty:
            for name in perf_data.columns: fig.add_trace(go.Scatter(x=perf_data.index, y=perf_data[name], name=name, mode='lines', line=dict(width=2)), secondary_y=False)
        if vix_data is not None: fig.add_trace(go.Scatter(x=vix_data.index, y=vix_data, name="VIX (Right Axis)", mode='lines', line=dict(width=1.5, color='rgba(255, 165, 0, 0.8)', dash='dot')), secondary_y=True)
        fig.update_layout(title=f"Market Indices Performance ({period})", template="plotly_dark", legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0), height=400, margin=dict(l=50, r=60, t=80, b=50), hovermode="x unified")
        fig.update_yaxes(title_text="% Change", secondary_y=False, zeroline=True, zerolinewidth=1, zerolinecolor='grey')
        if vix_data is not None: fig.update_yaxes(title_text="VIX Level", secondary_y=True, showgrid=False)
        logger.info("Successfully generated market overview chart.")
        return fig
    except Exception as e:
        logger.error(f"Error generating market overview chart: {str(e)}", exc_info=True)
        return None

# --- AI Response Function (from script 1 - context aware) ---
def get_ai_response(query, model, available_data=None):
    """Generate response from the selected LLM based on query and available data context."""
    if not model:
        logger.error("AI response requested but no model is selected or available.")
        return "AI model not available. Please select a model or check API key configuration."

    model_name = getattr(model, 'model_name', 'Unknown Model')
    logger.info(f"Generating AI response using model: {model_name} for query: '{query[:50]}...'")

    try:
        context = "Current data context:\n"; context_items = []
        if available_data and isinstance(available_data, dict):
            if not available_data: context += " - No stock data is currently loaded.\n"
            else:
                for symbol, stock_info in available_data.items():
                    if isinstance(stock_info, dict):
                        df = stock_info.get("data"); info = stock_info.get("info", {})
                        name = info.get("shortName", symbol); source = info.get("source", "Unknown")
                        item_str = f"- **{name} ({symbol})** (Source: {source}):"; details = []
                        if df is not None and not df.empty:
                            first = df.index.min().strftime('%Y-%m-%d'); last = df.index.max().strftime('%Y-%m-%d')
                            details.append(f"Data from {first} to {last}.")
                            if 'Close' in df.columns and pd.notna(df['Close'].iloc[-1]): details.append(f"Last close: ${df['Close'].iloc[-1]:.2f}.")
                        if 'sector' in info and pd.notna(info['sector']): details.append(f"Sector: {info['sector']}.")
                        if 'industry' in info and pd.notna(info['industry']): details.append(f"Industry: {info['industry']}.")
                        if 'marketCap' in info and pd.notna(info['marketCap']): details.append(f"MCap: ${info['marketCap']:,.0f}.")
                        if 'trailingPE' in info and pd.notna(info['trailingPE']): details.append(f"P/E: {info['trailingPE']:.2f}.")
                        if details: item_str += " " + " ".join(details)
                        else: item_str += " Metadata available but no price data loaded."
                        context_items.append(item_str)
                    else: context_items.append(f"- {symbol}: Error processing data.")
        else: context += " - No stock data context provided.\n"
        if context_items: context += "\n".join(sorted(context_items))

        prompt = f"""You are FinBot, a specialized financial analysis assistant. Provide objective, data-informed insights.
        **Context of Available Data:**
        ```
        {context if context_items else 'No specific stock data is currently loaded.'}
        ```
        **User Query:**
        ```
        {query}
        ```
        **Instructions:**
        1. Analyze query & context.
        2. Use provided data (source, dates, price, metadata) if relevant.
        3. Acknowledge limitations (real-time data, news, forecasts). Explain what you *can* do.
        4. Be objective & factual. Use neutral language.
        5. !!! NO INVESTMENT ADVICE !!! Do not tell user to buy/sell/hold. Discuss historicals, risks (beta), metrics (P/E) neutrally (e.g., "Historically...", "A high P/E might suggest...", "Factors often considered..."). Add disclaimer if query seems close to advice.
        6. Use clear structure (bullets, bold). Define terms if needed.
        7. Be concise.

        **Response:**
        """
        log_prompt_summary = f"Prompt for {model_name}: Context provided ({'Yes' if context_items else 'No'}). Query: '{query[:100]}...'"
        logger.info(log_prompt_summary)

        response = model.invoke(prompt)
        ai_message = response.content if hasattr(response, 'content') else str(response)
        ai_message = ai_message.strip()

        # Add disclaimer automatically if financial terms are detected and no disclaimer is present
        financial_keywords = ['buy', 'sell', 'hold', 'invest', 'advice', 'recommend', 'trade', 'profit', 'should i']
        response_lower = ai_message.lower()
        needs_disclaimer = any(keyword in query.lower() for keyword in financial_keywords) or \
                           any(keyword in response_lower for keyword in ['valuation', 'outlook', 'potential', 'undervalued', 'overvalued'])
        has_disclaimer = "investment advice" in response_lower or "not financial advice" in response_lower or "consult a professional" in response_lower

        if needs_disclaimer and not has_disclaimer:
             disclaimer = "\n\n*Disclaimer: I am an AI assistant and cannot provide financial advice. Information is for educational purposes only. Consult with a qualified financial professional before making investment decisions.*"
             ai_message += disclaimer
             logger.info("Added standard financial disclaimer to AI response.")

        return ai_message
    except Exception as e:
        logger.error(f"Error getting AI response from {model_name}: {str(e)}", exc_info=True)
        return f"Sorry, I encountered an error communicating with the AI model ({model_name}): {str(e)}"

# --- Connectivity Test Function (from script 1) ---
def test_api_connections():
    """Test connectivity to Yahoo Finance, Alpha Vantage, and Groq APIs"""
    test_results = []; logger.info("Starting API connection tests...")
    st.markdown("##### Testing Yahoo Finance")
    try:
        start = time.time(); stock = yf.Ticker("AAPL"); hist = stock.history(period="1d"); end = time.time()
        if not hist.empty: test_results.append(f"✅ Yahoo Finance (AAPL): Price ${hist['Close'].iloc[-1]:.2f} ({(end - start):.2f}s)")
        else: test_results.append(f"⚠ Yahoo Finance (AAPL): Connected but no data retrieved.")
    except Exception as e: test_results.append(f"❌ Yahoo Finance (AAPL): Connection failed - {str(e)}"); logger.error(f"YF test failed: {e}", exc_info=True)
    st.markdown("##### Testing Alpha Vantage")
    if ALPHA_VANTAGE_API_KEY and ALPHA_VANTAGE_API_KEY != "YOUR_ALPHA_VANTAGE_API_KEY":
        try:
            start = time.time(); params = {"function": "GLOBAL_QUOTE", "symbol": "MSFT", "apikey": ALPHA_VANTAGE_API_KEY}
            response = session.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=10); response.raise_for_status(); data = response.json(); end = time.time()
            if "Global Quote" in data and data["Global Quote"]: test_results.append(f"✅ Alpha Vantage (MSFT): Price ${data['Global Quote'].get('05. price', 'N/A')} ({(end - start):.2f}s)")
            elif "Note" in data: test_results.append(f"⚠ Alpha Vantage: API limit likely reached - {data['Note']}")
            elif "Error Message" in data: test_results.append(f"❌ Alpha Vantage: API Error - {data['Error Message']}")
            else: test_results.append(f"⚠ Alpha Vantage: Unexpected response: {str(data)[:100]}...")
        except requests.exceptions.RequestException as e: test_results.append(f"❌ Alpha Vantage: Connection failed - {str(e)}"); logger.error(f"AV connect failed: {e}", exc_info=True)
        except Exception as e: test_results.append(f"❌ Alpha Vantage: Test failed - {str(e)}"); logger.error(f"AV test failed: {e}", exc_info=True)
    else: test_results.append("ℹ️ Alpha Vantage: Skipped (No valid API key found).")
    st.markdown("##### Testing Groq AI")
    if GROQ_API_KEY:
        try:
            start = time.time(); test_model_name = "llama-3.1-8b-instant"; main_model_name = "llama-3.1-70b-versatile"
            model = None; response = None
            try: model = ChatGroq(model=test_model_name, temperature=0.1, api_key=GROQ_API_KEY, max_retries=1); response = model.invoke("Say hello.")
            except Exception: model = ChatGroq(model=main_model_name, temperature=0.1, api_key=GROQ_API_KEY, max_retries=1); response = model.invoke("Say hello.")
            end = time.time()
            if response and hasattr(response, 'content') and response.content: test_results.append(f"✅ Groq AI ({model.model_name}): Connected ({(end - start):.2f}s)")
            else: test_results.append(f"⚠ Groq AI ({model.model_name}): Connected but no response.")
        except Exception as e: test_results.append(f"❌ Groq AI: Connection failed - {str(e)}"); logger.error(f"Groq test failed: {e}", exc_info=True)
    else: test_results.append("ℹ️ Groq AI: Skipped (No API key found).")
    logger.info("API connection tests finished.")
    return test_results

# --- Utility Function ---
def format_number(num):
    """Format large numbers for display (from AV script)"""
    if num is None or isinstance(num, str) or pd.isna(num): return "N/A"
    try:
        num = float(num)
        if abs(num) >= 1e12: return f"${num / 1e12:.2f}T"
        if abs(num) >= 1e9: return f"${num / 1e9:.2f}B"
        if abs(num) >= 1e6: return f"${num / 1e6:.2f}M"
        return f"${num:,.2f}"
    except (ValueError, TypeError): return "N/A"

# --- Streamlit App Layout and Logic ---

st.title("💹 Stock Analysis & AI Assistant")

# --- Initialize Session State ---
if "messages" not in st.session_state: st.session_state.messages = []
if "stock_data" not in st.session_state: st.session_state.stock_data = {}
if "selected_symbols" not in st.session_state: st.session_state.selected_symbols = ["AAPL", "MSFT"]
if "models" not in st.session_state:
    st.session_state.models = initialize_models()
    logger.info(f"Models loaded into session state: {list(st.session_state.models.keys())}")
if "selected_model" not in st.session_state:
    available_model_keys = list(st.session_state.models.keys())
    st.session_state.selected_model = st.session_state.models.get(available_model_keys[0]) if available_model_keys else None
    logger.debug(f"Initialized session state: selected_model = {available_model_keys[0] if available_model_keys else 'None'}")
if "data_source" not in st.session_state:
    st.session_state.data_source = "Alpha Vantage" if (ALPHA_VANTAGE_API_KEY and ALPHA_VANTAGE_API_KEY != "YOUR_ALPHA_VANTAGE_API_KEY") else "Yahoo Finance"
    logger.debug(f"Initialized session state: data_source = {st.session_state.data_source}")
if "last_request_params" not in st.session_state: st.session_state.last_request_params = None
if "test_results" not in st.session_state: st.session_state.test_results = None
if "av_function" not in st.session_state: st.session_state.av_function = "TIME_SERIES_DAILY" # Default AV function
if "av_output_size" not in st.session_state: st.session_state.av_output_size = "compact"
if "av_interval" not in st.session_state: st.session_state.av_interval = "15min"


# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # --- API Keys ---
    st.subheader("API Keys")
    # Allow user to override AV key if needed
    user_av_key = st.text_input("Alpha Vantage API Key (Optional)", value=ALPHA_VANTAGE_API_KEY if ALPHA_VANTAGE_API_KEY != "YOUR_ALPHA_VANTAGE_API_KEY" else "", type="password", help="Overrides key from .env or default.")
    if user_av_key:
        ALPHA_VANTAGE_API_KEY = user_av_key # Use user input if provided
    if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_API_KEY":
         st.warning("Enter Alpha Vantage Key for AV source.", icon="🔑")

    # Check Groq Key
    if not GROQ_API_KEY:
         st.warning("Groq API Key missing (.env). AI disabled.", icon="🤖")

    # --- Data Source & Symbols ---
    st.subheader("Data Source")
    available_sources = ["Yahoo Finance"]
    if ALPHA_VANTAGE_API_KEY and ALPHA_VANTAGE_API_KEY != "YOUR_ALPHA_VANTAGE_API_KEY":
         available_sources.insert(0, "Alpha Vantage")
    else:
         if st.session_state.data_source == "Alpha Vantage": # If AV was selected but key is now invalid
             st.session_state.data_source = "Yahoo Finance" # Fallback
    available_sources.append("Saved Excel Files")

    try: current_source_index = available_sources.index(st.session_state.data_source)
    except ValueError: current_source_index = available_sources.index("Yahoo Finance"); st.session_state.data_source = "Yahoo Finance"

    st.session_state.data_source = st.radio(
        "Select Data Source:", available_sources, index=current_source_index, key="data_source_radio"
    )

    st.subheader("Stock Symbols")
    symbols_input = st.text_input("Enter symbols (comma-separated)", value=",".join(st.session_state.selected_symbols), key="symbols_input", placeholder="e.g., AAPL, MSFT, ^GSPC")
    current_symbols_in_textbox = sorted(list(set([s.strip().upper() for s in symbols_input.split(',') if s.strip()])))

    # --- Time Range / AV Parameters ---
    st.subheader("Time Range / Parameters")
    if st.session_state.data_source == "Alpha Vantage":
        # Use AV specific controls
        function_options_display = {"Intraday": "TIME_SERIES_INTRADAY", "Daily": "TIME_SERIES_DAILY", "Weekly": "TIME_SERIES_WEEKLY", "Monthly": "TIME_SERIES_MONTHLY"}
        selected_func_display = st.selectbox("AV Time Series:", list(function_options_display.keys()), index=1, key="av_func_select") # Default Daily
        st.session_state.av_function = function_options_display[selected_func_display]

        if st.session_state.av_function == "TIME_SERIES_INTRADAY":
            interval_options_display = {"1 min": "1min", "5 min": "5min", "15 min": "15min", "30 min": "30min", "60 min": "60min"}
            selected_interval_display = st.selectbox("AV Interval:", list(interval_options_display.keys()), index=2, key="av_interval_select") # Default 15min
            st.session_state.av_interval = interval_options_display[selected_interval_display]
        else:
            st.session_state.av_interval = None # Interval not applicable

        output_size_options_display = {"Compact (100 points)": "compact", "Full (All data)": "full"}
        selected_output_display = st.selectbox("AV Data Size:", list(output_size_options_display.keys()), index=0, key="av_output_select")
        st.session_state.av_output_size = output_size_options_display[selected_output_display]
        # Reset period/dates when AV is selected
        time_period = None; start_date = None; end_date = None
        st.caption("Time range is determined by AV parameters above.")
    elif st.session_state.data_source == "Yahoo Finance":
        # Use period/date controls for Yahoo
        time_options = ("1mo", "3mo", "6mo", "YTD", "1y", "2y", "5y", "max", "Custom")
        time_period = st.selectbox("Select Time Period:", time_options, index=4, key="yf_time_period_select") # Default 1y
        start_date = None; end_date = None
        if time_period == "Custom":
            col1, col2 = st.columns(2)
            with col1: start_date = st.date_input("Start Date", datetime.now().date() - pd.Timedelta(days=365), key="yf_start_date", max_value=datetime.now().date())
            with col2: end_date = st.date_input("End Date", datetime.now().date(), key="yf_end_date", max_value=datetime.now().date())
            if start_date > end_date: st.error("Start date cannot be after end date."); start_date, end_date = None, None
    else: # Saved Excel
        time_period = None; start_date = None; end_date = None
        st.caption("Time period depends on the saved file content.")


    # --- Fetch Data Button ---
    if st.button("🔄 Fetch / Load Data", key="fetch_data_button", type="primary", use_container_width=True):
        if not current_symbols_in_textbox:
            st.warning("Please enter at least one stock symbol.")
        elif st.session_state.data_source == "Alpha Vantage" and (not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_API_KEY"):
             st.error("Alpha Vantage source selected, but API Key is missing or invalid.")
        else:
            st.session_state.selected_symbols = current_symbols_in_textbox
            st.session_state.stock_data = {}
            st.session_state.messages = [] # Clear chat on new fetch
            logger.info(f"Fetch/Load requested for {st.session_state.selected_symbols} from {st.session_state.data_source}")
            # Store params relevant to the selected source
            request_params = {
                "symbols": st.session_state.selected_symbols,
                "source": st.session_state.data_source,
                "timestamp": time.time()
            }
            if st.session_state.data_source == "Alpha Vantage":
                request_params["av_function"] = st.session_state.av_function
                request_params["av_output_size"] = st.session_state.av_output_size
                request_params["av_interval"] = st.session_state.av_interval
            elif st.session_state.data_source == "Yahoo Finance":
                 request_params["period"] = time_period
                 request_params["start"] = start_date
                 request_params["end"] = end_date

            st.session_state.last_request_params = request_params
            logger.debug(f"Set last_request_params: {st.session_state.last_request_params}")
            st.rerun()


    # --- Chart Options ---
    st.subheader("📊 Chart Options")
    chart_type = st.selectbox("Chart Type", ["Candlestick", "Line", "OHLC"], key="chart_type_select", index=0)
    show_volume = st.checkbox("Show Volume", value=True, key="show_volume_check")
    show_ma = st.checkbox("Show Moving Averages", value=True, key="show_ma_check")
    ma_periods_input = "20, 50"
    if show_ma: ma_periods_input = st.text_input("MA Periods (e.g., 20,50,100)", value="20, 50", key="ma_periods_input")
    try: ma_periods = sorted(list(set([int(p.strip()) for p in ma_periods_input.split(',') if p.strip().isdigit() and int(p.strip()) > 0])))
    except ValueError: st.warning("Invalid MA periods. Using defaults (20, 50)."); ma_periods = [20, 50]
    show_bbands = st.checkbox("Show Bollinger Bands (20, 2)", value=False, key="show_bbands_check")


    # --- AI Model Selection ---
    st.subheader("🤖 AI Assistant")
    available_models_dict = st.session_state.get("models", {})
    available_model_names = list(available_models_dict.keys())
    if available_model_names and GROQ_API_KEY:
        current_model_object = st.session_state.get("selected_model")
        current_model_name = next((name for name, model in available_models_dict.items() if model == current_model_object), None)
        if current_model_name not in available_model_names: current_model_name = available_model_names[0]; st.session_state.selected_model = available_models_dict[current_model_name]
        try: current_model_index = available_model_names.index(current_model_name)
        except ValueError: current_model_index = 0
        selected_model_name = st.selectbox("Select LLM Model:", available_model_names, index=current_model_index, key="model_select")
        newly_selected_model_object = available_models_dict.get(selected_model_name)
        if newly_selected_model_object != st.session_state.selected_model: st.session_state.selected_model = newly_selected_model_object; logger.info(f"Switched AI model to: {selected_model_name}")
    elif not GROQ_API_KEY:
         st.info("Groq API Key needed for AI features.", icon="ℹ️")
         st.session_state.selected_model = None
    else: # Models failed to initialize
        st.error("LLM models failed to load. Check logs.", icon="🔥")
        st.session_state.selected_model = None

    # --- Connectivity Test ---
    st.subheader("📡 Connectivity")
    if st.button("Test API Connections", key="test_api_button", use_container_width=True):
        st.session_state.test_results = None; st.rerun()
    if st.session_state.get("test_results") is None and st.session_state.get("test_api_button", False):
        with st.spinner("Testing connections..."):
             st.session_state.test_results = test_api_connections()
             st.session_state.test_api_button = False # Reset button state
             st.rerun()

# --- Main Display Area ---

# Display API Test Results
if st.session_state.get("test_results"):
    st.subheader("API Connection Test Results")
    with st.expander("Show Test Details", expanded=True):
        results = st.session_state.test_results
        for result in results:
            if "✅" in result: st.success(result, icon="✅")
            elif "⚠" in result or "ℹ️" in result: st.warning(result, icon="⚠️")
            else: st.error(result, icon="❌")
    st.session_state.test_results = None # Clear after showing
    st.divider()

# --- Data Fetching and Loading Logic ---
if st.session_state.get("last_request_params"):
    params = st.session_state.last_request_params
    symbols_to_process = params["symbols"]
    source = params["source"]
    logger.info(f"Processing request triggered for {len(symbols_to_process)} symbols from {source}.")
    st.subheader(f"Acquiring Data: {', '.join(symbols_to_process)}")
    progress_bar = st.progress(0.0, text="Initializing...")
    status_text = st.empty()
    all_symbols_data = {}
    fetch_start_time = time.time()

    if source == "Saved Excel Files":
        status_text.text("Loading data from saved Excel files...")
        saved_data_result = get_data_from_excel_files(symbols_to_process)
        all_symbols_data = saved_data_result.get("data", {})
        load_message = saved_data_result.get("text", "Finished loading.")
        if not all_symbols_data: status_text.warning(load_message or "Could not load any data.")
        else: status_text.success(load_message)
        progress_bar.progress(1.0, text="Loading complete.")
    else: # Fetching from API (AV or YF)
        total_symbols = len(symbols_to_process)
        symbols_processed_success = 0
        for i, symbol in enumerate(symbols_to_process):
            current_progress = (i + 1) / total_symbols
            status_text.text(f"Fetching data for {symbol} ({i+1}/{total_symbols})...")
            progress_bar.progress(current_progress, text=f"Fetching {symbol}...")
            logger.info(f"Fetching data for symbol: {symbol} from {source}")
            df, info_or_error = None, None
            try:
                if source == "Alpha Vantage":
                    # Get parameters from stored request
                    av_func = params.get("av_function", "TIME_SERIES_DAILY")
                    av_output = params.get("av_output_size", "compact")
                    av_interval = params.get("av_interval") # Can be None
                    df, info_or_error = get_alpha_vantage_data(symbol, av_func, av_output, av_interval)
                    # Add small delay for AV free tier
                    if total_symbols > 1 and i < total_symbols - 1:
                         time.sleep(12.5) # ~5 requests per minute allowed

                elif source == "Yahoo Finance":
                    yf_period = params.get("period")
                    yf_start = params.get("start")
                    yf_end = params.get("end")
                    df, info_or_error = get_yahoo_finance_data(symbol, yf_period, yf_start, yf_end)

                # --- Process Result for the Symbol ---
                if isinstance(info_or_error, str): # Error message returned
                     st.error(f"Error for {symbol}: {info_or_error}")
                     logger.error(f"Failed {symbol}: {info_or_error}")
                     continue # Skip to next symbol
                elif df is None or df.empty:
                     warn_msg = f"No price data found for {symbol}."
                     if isinstance(info_or_error, dict) and info_or_error: warn_msg += " Company info might be available."
                     st.warning(warn_msg, icon="⚠️")
                     logger.warning(f"No price data for {symbol}. Info: {'Dict received' if isinstance(info_or_error, dict) else 'Error/None'}")
                     # Store info even if data is missing
                     all_symbols_data[symbol] = {"data": pd.DataFrame(), "info": info_or_error if isinstance(info_or_error, dict) else {"symbol": symbol, "source": source}}
                else: # Data received successfully
                     info_dict = info_or_error if isinstance(info_or_error, dict) else {"symbol": symbol, "shortName": symbol, "source": source}
                     all_symbols_data[symbol] = {"data": df, "info": info_dict}
                     logger.info(f"Successfully processed data for {symbol} from {source}. Rows: {len(df)}")
                     symbols_processed_success += 1

            except Exception as fetch_err:
                error_message = f"Failed fetch/process for {symbol}: {fetch_err}"
                st.error(error_message); logger.error(error_message, exc_info=True)

    fetch_end_time = time.time()
    st.session_state.stock_data = all_symbols_data
    st.session_state.last_request_params = None # Clear request params
    logger.debug("Cleared last_request_params after processing.")

    if not st.session_state.stock_data: status_text.warning("No data loaded/fetched for any symbol.", icon="⚠️")
    else: status_text.success(f"Data acquired for {symbols_processed_success}/{len(symbols_to_process)} symbols in {fetch_end_time - fetch_start_time:.2f}s.")
    # Use toast for completion message if needed, then rerun to clear progress
    st.toast("Data update complete!", icon="✅")
    time.sleep(1.5) # Give toast time to show
    st.rerun()

# --- Display Market Overview Chart ---
st.subheader("Market Overview (Yahoo Finance)")
market_overview_period = st.selectbox("Select Market Chart Period:", ("1mo", "3mo", "6mo", "YTD", "1y", "2y", "5y"), index=2, key="market_overview_period_select")
market_fig = generate_market_overview_chart(period=market_overview_period)
if market_fig: st.plotly_chart(market_fig, use_container_width=True)
else: st.info("Could not generate market overview chart.")
st.divider()

# --- Display Data and Charts for Each Loaded Symbol ---
if not st.session_state.stock_data:
    st.info("No stock data loaded. Use the sidebar to fetch or load data.", icon="ℹ️")
else:
    st.subheader("Stock Details & Charts")
    sorted_symbols = sorted(st.session_state.stock_data.keys())

    for symbol in sorted_symbols:
        stock_info_entry = st.session_state.stock_data[symbol]
        data = stock_info_entry.get("data")
        info = stock_info_entry.get("info", {})
        company_name = info.get("shortName", symbol)

        st.markdown(f"#### {company_name} ({symbol})")

        # --- Display Key Metrics & Save Button ---
        cols = st.columns([1.5, 1.5, 1.5, 2]) # Adjusted widths

        with cols[0]: # Last Price / Current Price
            last_close = data['Close'].iloc[-1] if data is not None and not data.empty and 'Close' in data.columns else np.nan
            current_price_info = info.get('currentPrice') # From AV Quote
            display_price = np.nan
            label = "Price"
            if pd.notna(current_price_info): display_price = current_price_info; label = "Current Price"
            elif pd.notna(last_close): display_price = last_close; label = "Last Close"
            st.metric(label=label, value=f"${display_price:.2f}" if pd.notna(display_price) else "N/A")

        with cols[1]: # Change
            change_info = info.get('change')
            change_pct_str = info.get('changePercent', '')
            delta_val = None; display_change = np.nan
            if pd.notna(change_info) and change_pct_str:
                 display_change = change_info
                 try: change_pct = float(change_pct_str.replace('%',''))
                 except: change_pct = np.nan
                 delta_val = f"{change_info:+.2f} ({change_pct:+.2f}%)" if pd.notna(change_pct) else f"{change_info:+.2f}"
            elif data is not None and not data.empty and 'Close' in data.columns and len(data) >= 2: # Fallback daily change
                 last_c = data['Close'].iloc[-1]; prev_c = data['Close'].iloc[-2]
                 if pd.notna(last_c) and pd.notna(prev_c) and prev_c != 0:
                     change_hist = last_c - prev_c; change_pct_hist = (change_hist / prev_c) * 100
                     display_change = change_hist
                     delta_val = f"{change_hist:+.2f} ({change_pct_hist:+.2f}%)"
            st.metric(label="Change", value=f"{display_change:+.2f}" if pd.notna(display_change) else "N/A", delta=delta_val)

        with cols[2]: # Volume
             volume_info = info.get('volume') # From AV Quote
             last_vol_data = data['Volume'].iloc[-1] if data is not None and not data.empty and 'Volume' in data.columns else np.nan
             display_vol = np.nan; label = "Volume"
             if pd.notna(volume_info): display_vol = volume_info
             elif pd.notna(last_vol_data): display_vol = last_vol_data; label = "Last Volume"
             st.metric(label=label, value=f"{display_vol:,.0f}" if pd.notna(display_vol) else "N/A")

        with cols[3]: # Save Button
             if data is not None and not data.empty:
                 st.write("") # Align button vertically
                 filename, excel_bytes = save_data_to_excel(symbol, data, info)
                 if filename and excel_bytes:
                     st.download_button(label=f"💾 Save {symbol} (.xlsx)", data=excel_bytes, file_name=filename,
                                         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"download_{symbol}")

        # --- Display Chart ---
        chart_fig = generate_stock_chart(data, info, chart_type, show_ma, ma_periods, show_volume, show_bbands)
        if chart_fig: st.plotly_chart(chart_fig, use_container_width=True)
        else: st.warning(f"Could not generate chart for {symbol}.", icon="📊")

        # --- Display Company Info / Summary ---
        with st.expander("Company Information & Data Summary", expanded=False):
            if info:
                info_cols = st.columns(2)
                with info_cols[0]:
                    st.write(f"**Name:** {info.get('shortName', symbol)}")
                    st.write(f"**Symbol:** {info.get('symbol', symbol)}")
                    st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                    st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                    st.write(f"**Data Source:** {info.get('source', 'N/A')}")
                    if 'latestTradingDay' in info: st.write(f"**Latest Quote Day:** {info['latestTradingDay']}")
                with info_cols[1]:
                    st.write(f"**Market Cap:** {format_number(info.get('marketCap'))}") # Use format_number
                    st.write(f"**P/E Ratio (TTM):** {f'{float(info.get("trailingPE")):.2f}' if pd.notna(info.get('trailingPE')) and isinstance(info.get('trailingPE'), (int, float)) else 'N/A'}")
                    div_yield = info.get('dividendYield')
                    st.write(f"**Dividend Yield:** {f'{float(div_yield)*100:.2f}%' if pd.notna(div_yield) and isinstance(div_yield, (int, float)) and div_yield > 0 else 'N/A'}")
                    st.write(f"**Beta:** {f'{float(info.get("beta")):.2f}' if pd.notna(info.get("beta")) and isinstance(info.get("beta"), (int, float)) else 'N/A'}")
                    low52 = info.get('fiftyTwoWeekLow'); high52 = info.get('fiftyTwoWeekHigh')
                    if pd.notna(low52) and pd.notna(high52): st.write(f"**52-Wk Range:** {format_number(low52)} - {format_number(high52)}")
                summary = info.get('longBusinessSummary')
                if summary: st.write("**Business Summary:**"); st.markdown(f"> {summary}")
                else: st.write("**Business Summary:** Not available.")
            if data is not None and not data.empty: st.write("**Data Summary (Last 5 rows):**"); st.dataframe(data.tail().round(2))
            else: st.write("**Data:** No price data available.")
        st.divider()


# --- Chat Interface ---
st.subheader("🤖 AI Financial Assistant")

# Welcome message if chat is empty
if not st.session_state.messages:
     st.markdown("""
     <div class="info-card">
         <p>Ask me about the loaded stock data, market trends, financial concepts, or anything else!</p>
         <p><strong>Tip:</strong> Ask questions like "Summarize the recent performance of AAPL based on the loaded data" or "Explain what P/E ratio means for MSFT".</p>
         <p><strong>Current AI Model:</strong> {model_name}</p>
     </div>
     """.format(model_name=getattr(st.session_state.selected_model, 'model_name', 'N/A') if st.session_state.selected_model else "No model available"), unsafe_allow_html=True)


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your question..." + (" (AI Disabled)" if not st.session_state.selected_model else "")):
    if not st.session_state.selected_model:
        st.warning("AI Assistant is disabled. Please check Groq API Key and model selection.", icon="🤖")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        logger.info(f"User prompt: '{prompt}'")
        with st.chat_message("assistant"):
            message_placeholder = st.empty(); message_placeholder.markdown("Thinking...")
            try:
                # Use the context-aware AI function
                ai_response = get_ai_response(prompt, st.session_state.selected_model, st.session_state.stock_data)
                message_placeholder.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                logger.info(f"AI response generated (length: {len(ai_response)})")
            except Exception as ai_error:
                error_msg = f"Error during AI processing: {ai_error}"
                logger.error(error_msg, exc_info=True)
                message_placeholder.error(f"Sorry, an error occurred: {ai_error}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {ai_error}"})

# Clear Chat Button
if len(st.session_state.messages) > 0:
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        logger.info("Chat history cleared by user.")
        st.rerun()

logger.info("--- Streamlit script execution finished ---")