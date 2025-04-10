import streamlit as st
import yfinance as yf
import datetime
import pandas as pd
import numpy as np
import time
import traceback  # For better error reporting
from langchain_groq import ChatGroq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Add this function to test the yfinance connectivity
def test_yfinance_connection():
    """Test if yfinance can fetch data properly"""
    test_tickers = ["AAPL", "^GSPC"]
    test_results = []
    
    st.markdown("### Testing Yahoo Finance Connection")
    
    for ticker in test_tickers:
        try:
            start_time = time.time()
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1d")
            end_time = time.time()
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                test_results.append(f"✅ {ticker}: ${current_price:.2f} (fetched in {(end_time - start_time):.2f}s)")
            else:
                test_results.append(f"⚠️ {ticker}: No data retrieved (API connected but no data)")
        except Exception as e:
            test_results.append(f"❌ {ticker}: Connection failed - {str(e)}")
    
    return test_results

# Initialize different LLM models
def initialize_models():
    models = {}
    
    # Check for GROQ_API_KEY
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        groq_api_key = "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"
    
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
    except Exception as e:
        st.error(f"⚠️ Error initializing models: {str(e)}")
        return {}
    
    return models

# Function to generate stock chart
def generate_stock_chart(symbol, period="1mo"):
    """Generate an interactive stock chart for the given symbol"""
    try:
        # Get historical data
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if hist.empty:
            return None
            
        # Get company name
        name = stock.info.get('shortName', symbol)
        
        # Create subplot figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name="Price"
            )
        )
        
        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=hist.index,
                y=hist['Volume'],
                name="Volume",
                marker_color='rgba(100, 100, 200, 0.4)'
            ),
            secondary_y=True
        )
        
        # Add moving averages
        if len(hist) >= 20:
            ma20 = hist['Close'].rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=ma20,
                    name="20-day MA",
                    line=dict(color='orange', width=1.5)
                )
            )
            
        if len(hist) >= 50:
            ma50 = hist['Close'].rolling(window=50).mean()
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=ma50,
                    name="50-day MA",
                    line=dict(color='red', width=1.5)
                )
            )
        
        # Set chart title and axis labels
        fig.update_layout(
            title=f"{name} ({symbol}) - Stock Price Chart",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            yaxis2_title="Volume",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
            margin=dict(l=50, r=50, t=80, b=50),
        )
        
        # Update y-axis range for volume
        fig.update_yaxes(
            range=[0, hist['Volume'].max() * 3],
            secondary_y=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error generating chart: {str(e)}")
        return None

# Function to generate market overview chart
def generate_market_overview_chart(period="1mo"):
    """Generate a comparison chart of major market indices"""
    try:
        indices = {
            "S&P 500": "^GSPC",
            "Dow Jones": "^DJI",
            "NASDAQ": "^IXIC"
        }
        
        # Download historical data for all indices
        data = {}
        for name, symbol in indices.items():
            index = yf.Ticker(symbol)
            hist = index.history(period=period)
            if not hist.empty:
                # Normalize to percentage change from first day
                first_price = hist['Close'].iloc[0]
                data[name] = (hist['Close'] / first_price - 1) * 100
        
        if not data:
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Add each index as a line
        for name, values in data.items():
            fig.add_trace(
                go.Scatter(
                    x=values.index,
                    y=values,
                    name=name,
                    mode='lines',
                    line=dict(width=2)
                )
            )
            
        # Configure layout
        fig.update_layout(
            title="Market Indices Comparison (% Change)",
            xaxis_title="Date",
            yaxis_title="% Change",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400,
            margin=dict(l=50, r=50, t=80, b=50),
        )
        
        # Add zero line
        fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="white")
        
        return fig
        
    except Exception as e:
        st.error(f"Error generating market overview chart: {str(e)}")
        return None

# Set page config with custom theme
st.set_page_config(
    page_title="Stock Market Analysis Chatbot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with dark theme
st.markdown("""
    <style>
    .main {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    .stApp {
        background-color: #1e1e1e;
    }
    .css-1d391kg, .css-1wrcr25 {
        background-color: #1e1e1e;
    }
    .stChatMessage {
        background-color: #2d2d2d;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        color: #e0e0e0;
    }
    .stChatInput {
        background-color: #2d2d2d;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        color: #e0e0e0;
        border: 1px solid #3d3d3d;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 20px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1668a3;
    }
    .stock-card {
        background-color: #2d2d2d;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .css-1offfwp {
        color: #e0e0e0;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #e0e0e0;
    }
    .css-81oif8 {
        color: #e0e0e0;
    }
    .css-pkbazv {
        color: #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

# Function to extract finance keywords from text
def extract_finance_keywords(text):
    """Extract potential stock symbols and finance terms from a question"""
    # Common stock symbols mapping
    stock_name_to_symbol = {
        "APPLE": "AAPL", 
        "MICROSOFT": "MSFT", 
        "GOOGLE": "GOOGL", 
        "ALPHABET": "GOOGL",
        "AMAZON": "AMZN", 
        "META": "META", 
        "FACEBOOK": "META",
        "TESLA": "TSLA", 
        "NVIDIA": "NVDA",
        "NETFLIX": "NFLX",
        "BERKSHIRE": "BRK-B",
        "JPMORGAN": "JPM",
        "WALMART": "WMT",
        "DISNEY": "DIS",
        "COCA-COLA": "KO",
        "INTEL": "INTC",
        "AMD": "AMD"
    }
    
    # Market indices
    indices = {
        "S&P": "^GSPC",
        "S&P 500": "^GSPC", 
        "DOW": "^DJI",
        "DOW JONES": "^DJI",
        "NASDAQ": "^IXIC",
        "RUSSELL": "^RUT",
        "VIX": "^VIX"
    }
    
    words = text.split()
    keywords = []
    symbols = []
    
    # Common financial terms to look for
    finance_terms = [
        "stock", "market", "nasdaq", "dow", "s&p", "bond", "etf", "fund", 
        "dividend", "invest", "share", "price", "earning", "portfolio", 
        "trade", "revenue", "profit", "loss", "growth", "recession"
    ]
    
    for word in words:
        # Clean the word
        clean_word = word.strip('.,;:!?()[]{}"\'-').upper()
        
        # Check if it looks like a stock symbol (1-5 uppercase letters)
        if clean_word.isalpha() and 1 <= len(clean_word) <= 5 and clean_word.isupper():
            symbols.append(clean_word)
        
        # Check for finance terms
        for term in finance_terms:
            if term.lower() in word.lower() and term not in keywords:
                keywords.append(term)
    
    # Look for company names
    for company, symbol in stock_name_to_symbol.items():
        if company in text.upper() and symbol not in symbols:
            symbols.append(symbol)
    
    # Look for market indices
    for index_name, symbol in indices.items():
        if index_name.upper() in text.upper() and symbol not in symbols:
            symbols.append(symbol)
    
    # If no symbols found but market-related terms exist, include major indices
    if not symbols and any(term in ["MARKET", "STOCKS", "STOCK MARKET"] for term in text.upper().split()):
        symbols = ["^GSPC", "^DJI", "^IXIC"]  # S&P 500, Dow Jones, NASDAQ
    
    return {"keywords": keywords, "symbols": symbols}

# Function to fetch real-time stock data
def fetch_finance_data(symbols, debug=False):
    """Fetch real-time stock data for given symbols using yfinance with improved error handling"""
    if not symbols:
        return {"text": "No specific stock symbols identified in your question.", "charts": {}}
    
    if debug:
        st.write(f"Attempting to fetch data for symbols: {symbols}")
        
    results = []
    charts = {}
    today = datetime.datetime.now().date()
    
    # Handle up to 5 symbols to avoid overloading
    for symbol in symbols[:5]:
        try:
            if debug:
                st.write(f"Fetching data for {symbol}...")
                
            # Get the stock info
            start_time = time.time()
            stock = yf.Ticker(symbol)
            
            # Force a fresh download to avoid cached data
            hist = stock.history(period="2d", proxy=None, rounding=True)
            info = stock.info
            end_time = time.time()
            
            if debug:
                st.write(f"Data fetch time: {end_time - start_time:.2f} seconds")
                st.write(f"Data points retrieved: {len(hist)}")
            
            if hist.empty:
                results.append(f"No data available for {symbol}")
                continue
                
            # Basic stock info
            name = info.get('shortName', symbol)
            current_price = hist['Close'].iloc[-1] if not hist.empty else "N/A"
            
            # Calculate price change
            if len(hist) >= 2:
                prev_close = hist['Close'].iloc[-2]
                price_change = current_price - prev_close
                change_percent = (price_change / prev_close) * 100
                change_str = f"{price_change:.2f} ({change_percent:.2f}%)"
                change_direction = "🔼" if price_change >= 0 else "🔽"
            else:
                change_str = "N/A"
                change_direction = ""
            
            # Get additional metrics when available
            market_cap = info.get('marketCap', "N/A")
            if market_cap not in ["N/A", None]:
                if market_cap > 1000000000:
                    market_cap = f"${market_cap / 1000000000:.2f}B"
                else:
                    market_cap = f"${market_cap / 1000000:.2f}M"
            
            # Add volume information
            volume = hist['Volume'].iloc[-1] if 'Volume' in hist and not hist.empty else "N/A"
            if volume != "N/A":
                volume_display = f"{volume:,.0f}"
            else:
                volume_display = "N/A"
                
            # Store chart info
            charts[symbol] = {
                "symbol": symbol,
                "name": name
            }
                
            # Format the result
            result = f"""
**{name} ({symbol})**: ${current_price:.2f} {change_direction} {change_str}
**Market Cap**: {market_cap}
**Volume**: {volume_display}
**Data as of**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
            results.append(result)
            
        except Exception as e:
            error_details = traceback.format_exc() if debug else str(e)
            results.append(f"Error fetching data for {symbol}: {str(e)}")
            if debug:
                st.error(f"Detailed error for {symbol}: {error_details}")
    
    return {"text": "\n".join(results), "charts": charts}

# Function to get market overview
def get_market_overview(debug=False):
    """Get summary of major market indices with improved error handling"""
    indices = {
        "S&P 500": "^GSPC",
        "Dow Jones": "^DJI",
        "NASDAQ": "^IXIC"
    }
    
    results = ["**Current Market Overview:**"]
    has_market_data = False
    
    for name, symbol in indices.items():
        try:
            if debug:
                st.write(f"Fetching market data for {name} ({symbol})...")
                
            index = yf.Ticker(symbol)
            
            # Force a fresh download with no caching
            hist = index.history(period="2d", proxy=None)
            
            if hist.empty:
                results.append(f"{name}: No data available")
                continue
                
            current = hist['Close'].iloc[-1]
            has_market_data = True
            
            if len(hist) >= 2:
                prev = hist['Close'].iloc[-2]
                change = current - prev
                percent = (change / prev) * 100
                direction = "🔼" if change >= 0 else "🔽"
                
                results.append(f"{name}: {current:.2f} {direction} {change:.2f} ({percent:.2f}%)")
            else:
                results.append(f"{name}: {current:.2f} (no previous close data)")
                
        except Exception as e:
            error_details = traceback.format_exc() if debug else str(e)
            results.append(f"{name}: Error fetching data - {str(e)}")
            if debug:
                st.error(f"Detailed error for {name}: {error_details}")
    
    return {"text": "\n".join(results), "has_data": has_market_data}

# Function to get model response
def get_model_response(question, model, debug=False):
    # Extract finance keywords
    finance_data = extract_finance_keywords(question)
    keywords = finance_data["keywords"]
    symbols = finance_data["symbols"]
    
    if debug:
        st.write("Extracted Keywords:", keywords)
        st.write("Detected Symbols:", symbols)
    
    # Prepare finance context
    finance_context = ""
    charts_to_display = {}
    show_market_chart = False
    
    # Get stock data if symbols were found
    if symbols:
        if debug:
            st.write("Fetching real-time stock data...")
        stock_data = fetch_finance_data(symbols, debug=debug)
        finance_context += "REAL-TIME STOCK DATA:\n\n"
        finance_context += stock_data["text"]
        finance_context += "\n\n"
        charts_to_display.update(stock_data["charts"])
    
    # Add market overview for general market questions
    if any(term in keywords for term in ["market", "stock", "nasdaq", "dow", "s&p"]):
        if debug:
            st.write("Fetching market overview...")
        market_data = get_market_overview(debug=debug)
        finance_context += market_data["text"]
        finance_context += "\n\n"
        show_market_chart = market_data["has_data"]
    
    # Create a detailed prompt for the model
    prompt = f"""You are an expert financial advisor and stock market analyst. Answer the user's question about stocks or market trends.
    
    USER QUESTION: {question}
    
    {finance_context}
    
    Your task:
    1. Analyze the user's question about finances, stocks, or market trends
    2. Use the real-time finance data provided above in your analysis
    3. Provide concise, factual analysis based on the latest data
    4. If recommending investment actions, include appropriate disclaimers
    
    Important: Base your analysis primarily on the current data provided above. Make your response conversational but focused on answering the user's specific question.
    """
    
    try:
        # Use the modern invoke method on ChatGroq models
        response = model.invoke(prompt)
        
        # Return both the response and chart info
        return {
            "content": response.content,
            "charts": charts_to_display,
            "show_market_chart": show_market_chart
        }
    except Exception as e:
        st.error(f"Error with model response: {str(e)}")
        return {
            "content": f"I encountered an error while processing your request. Error details: {str(e)}",
            "charts": {},
            "show_market_chart": False
        }

# Initialize models
available_models = initialize_models()

# Sidebar
with st.sidebar:
    st.title("📊 Stock Market Chatbot")
    
    # Model Selection
    st.markdown("### Select AI Model")
    if not available_models:
        st.error("No models available. Please check your API key configuration.")
        selected_model = None
    else:
        selected_model = st.selectbox(
            "Choose the AI model to use:",
            options=list(available_models.keys()),
            index=0
        )
    
    # Chart options
    st.markdown("### Chart Options")
    chart_period = st.selectbox(
        "Chart Time Period:",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        index=0,
        help="Select the time period for stock charts"
    )
    
    # Debug mode toggle
    debug_mode = st.checkbox("Debug Mode", value=False, help="Show detailed information about data fetching")
    
    # Test yfinance connection button
    if st.button("Test Yahoo Finance Connection"):
        connection_status = test_yfinance_connection()
        for status in connection_status:
            st.write(status)
    
    st.markdown("### Features")
    st.markdown("""
    - Stock market analysis with real-time data
    - Interactive stock price charts
    - Market trend visualization
    - Company insights
    - Financial guidance
    """)
    
    st.markdown("---")
    st.markdown("""
    ### Example Questions
    - What is the current price of AAPL?
    - How is the Nasdaq performing today?
    - Tell me about Tesla stock
    - What factors affect tech stock performance?
    - Show me a comparison of MSFT and GOOGL
    """)

# Main content
st.title("📈 Stock Market Analysis Assistant")
st.markdown(f"""
<div class="stock-card">
    <h3>Welcome to your AI-powered Stock Market Assistant!</h3>
    <p>Ask me anything about stocks, market trends, or company analysis. I can help you with:</p>
    <ul>
        <li>Real-time stock market data</li>
        <li>Market trend analysis</li>
        <li>Company performance insights</li>
        <li>Investment concepts and strategies</li>
    </ul>
    <p><strong>Current Model:</strong> {selected_model if selected_model else "No model available"}</p>
    <p><em>Note: This assistant provides real-time stock data through Yahoo Finance integration. However, this is not financial advice.</em></p>
</div>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Chat interface
user_input = st.chat_input("Ask about any stock or market analysis...")

# Process user input
if user_input and selected_model:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display the latest user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Display a placeholder for the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Use the selected Groq model for the response
                model = available_models[selected_model]
                response_data = get_model_response(user_input, model, debug=debug_mode)
                
                # Extract components
                response_text = response_data["content"]
                charts = response_data["charts"]
                show_market_chart = response_data["show_market_chart"]
                
                # Display the text response
                st.markdown(response_text, unsafe_allow_html=True)
                
                # Display stock charts if available
                if charts:
                    st.markdown("### Stock Price Charts")
                    for symbol, chart_info in charts.items():
                        with st.expander(f"📈 {chart_info['name']} ({symbol}) Chart", expanded=True):
                            chart = generate_stock_chart(symbol, period=chart_period)
                            if chart:
                                st.plotly_chart(chart, use_container_width=True)
                            else:
                                st.warning(f"Could not generate chart for {symbol}")
                
                # Display market overview chart if relevant
                if show_market_chart:
                    st.markdown("### Market Performance")
                    market_chart = generate_market_overview_chart(period=chart_period)
                    if market_chart:
                        st.plotly_chart(market_chart, use_container_width=True)
                
                # Add full response to chat history (without charts)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
            except Exception as e:
                error_msg = f"Error getting response from model: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}. Please try again."})