import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

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
    except Exception as e:
        st.error(f"⚠️ Error initializing models: {str(e)}")
        return {}
    
    return models

# Set page config with custom theme
st.set_page_config(
    page_title="Stock Market Analysis Chatbot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .stChatMessage {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stChatInput {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Function to get stock data
def get_stock_data(symbol, period="1mo"):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        if hist.empty:
            return None
        return hist
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None

# Function to create stock chart
def create_stock_chart(df, symbol):
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Close'])])
    fig.update_layout(
        title=f'{symbol} Stock Price',
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_white',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig

# Function to get company info
def get_company_info(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return info
    except Exception as e:
        st.error(f"Error fetching company info: {str(e)}")
        return None

# Function to analyze stock data
def analyze_stock(symbol):
    data = get_stock_data(symbol)
    if data is None or data.empty:
        return f"Sorry, I couldn't find data for {symbol}."
    
    current_price = data['Close'].iloc[-1]
    price_change = data['Close'].iloc[-1] - data['Close'].iloc[0]
    price_change_pct = (price_change / data['Close'].iloc[0]) * 100
    
    # Calculate some additional metrics
    avg_volume = data['Volume'].mean()
    high_52w = data['High'].max()
    low_52w = data['Low'].min()
    
    # Get company info
    company_info = get_company_info(symbol)
    company_name = company_info.get('longName', symbol) if company_info else symbol
    
    analysis = f"""
    <div class="stock-card">
        <h3>Analysis for {company_name} ({symbol})</h3>
        <p>Current Price: ${current_price:.2f}</p>
        <p>Price Change: ${price_change:.2f} ({price_change_pct:.2f}%)</p>
        <p>Average Volume: {int(avg_volume):,}</p>
        <p>52-Week High: ${high_52w:.2f}</p>
        <p>52-Week Low: ${low_52w:.2f}</p>
    </div>
    """
    return analysis

# Function to get response from selected model
def get_model_response(question, model):
    # Check for stock ticker symbol commands to handle locally
    if question.startswith("/analyze "):
        symbol = question.replace("/analyze ", "").strip().upper()
        return f"Here's the analysis for {symbol}:\n\n{analyze_stock(symbol)}"
    
    # Extract potential stock symbols from the question
    words = question.split()
    potential_symbols = [word.strip('.,;:!?()[]{}"\'-').upper() for word in words 
                if word.strip('.,;:!?()[]{}"\'-').isalpha() and 
                len(word.strip('.,;:!?()[]{}"\'-')) <= 5 and 
                word.strip('.,;:!?()[]{}"\'-').isupper()]
    
    # Also check for common stock patterns
    common_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD", "INTC", "JPM", "BAC", "GS"]
    for ticker in common_tickers:
        if ticker in question.upper() and ticker not in potential_symbols:
            potential_symbols.append(ticker)
    
    # Prepare context with stock data if symbols are detected
    context = ""
    if potential_symbols:
        context += "Here is the latest market data:\n\n"
        
    for symbol in potential_symbols:
        data = get_stock_data(symbol, period=default_period)
        if data is not None and not data.empty:
            # Get detailed stock information
            current_price = data['Close'].iloc[-1]
            prev_close = data['Close'].iloc[-2] if len(data) > 1 else data['Open'].iloc[-1]
            daily_change = current_price - prev_close
            daily_change_pct = (daily_change / prev_close) * 100
            
            period_change = data['Close'].iloc[-1] - data['Close'].iloc[0]
            period_change_pct = (period_change / data['Close'].iloc[0]) * 100
            
            # Get additional metrics
            avg_volume = data['Volume'].mean()
            high = data['High'].max()
            low = data['Low'].min()
            
            # Get company info
            company_info = get_company_info(symbol)
            company_name = company_info.get('longName', symbol) if company_info else symbol
            
            # Add comprehensive data to context
            context += f"- {company_name} ({symbol}):\n"
            context += f"  Current: ${current_price:.2f}\n"
            context += f"  Daily Change: ${daily_change:.2f} ({daily_change_pct:.2f}%)\n"
            context += f"  {default_period} Change: ${period_change:.2f} ({period_change_pct:.2f}%)\n"
            context += f"  Period High/Low: ${high:.2f}/${low:.2f}\n"
            context += f"  Avg Volume: {int(avg_volume):,}\n\n"
    
    # Create a detailed and explicit prompt for the model
    prompt = f"""You are an expert financial advisor and stock market analyst. Answer the user's question about stocks or market trends.
    
    USER QUESTION: {question}
    
    AVAILABLE REAL-TIME STOCK DATA:
    {context if context else "No specific stock data was found for any tickers mentioned in the question."}
    
    Your task:
    1. Analyze the user's question and the provided real-time stock data
    2. Use the real-time stock data in your analysis - BE SPECIFIC about current prices, changes, and trends
    3. If the user asked about a stock ticker that has data above, directly reference that data
    4. Provide concise, factual analysis based on the real-time data
    5. If recommending investment actions, include appropriate disclaimers
    
    Important: Base your analysis on the REAL-TIME DATA provided above, not on your general knowledge which may be outdated.
    """
    
    try:
        # Use the modern invoke method on ChatGroq models
        response = model.invoke(prompt)
        return response.content
    except Exception as e:
        st.error(f"Error with model response: {str(e)}")
        return f"I encountered an error while processing your request. Error details: {str(e)}"

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
    
    # Time period selection
    st.markdown("### Data Settings")
    default_period = st.selectbox(
        "Default time period for charts:",
        options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        index=2  # Default to 1 month
    )
    
    st.markdown("### Features")
    st.markdown("""
    - Real-time stock data
    - Interactive charts
    - Market analysis
    - Company insights
    """)
    st.markdown("---")
    st.markdown("""
    ### Example Commands
    - /analyze AAPL - Quick analysis of Apple stock
    - What's the current price of AAPL?
    - Show me the performance of MSFT
    - Analyze GOOGL's market sentiment
    - Compare AAPL and MSFT
    """)

# Main content
st.title("📈 Stock Market Analysis Assistant")
st.markdown("""
<div class="stock-card">
    <h3>Welcome to your AI-powered Stock Market Assistant!</h3>
    <p>Ask me anything about stocks, market trends, or company analysis. I can help you with:</p>
    <ul>
        <li>Real-time stock prices and charts</li>
        <li>Market trend analysis</li>
        <li>Company performance metrics</li>
        <li>Stock comparisons</li>
    </ul>
    <p><strong>Current Model:</strong> {}</p>
</div>
""".format(selected_model if selected_model else "No model available"), unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)
        
        # If the message contains a stock symbol, try to show a chart
        if message["role"] == "assistant":
            # Check for stock symbols
            words = message["content"].split()
            for word in words:
                # Clean the word of punctuation
                clean_word = word.strip('.,;:!?()')
                if clean_word.upper() in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"] or \
                   (clean_word.isupper() and len(clean_word) <= 5 and clean_word.isalpha()):
                    symbol = clean_word.upper()
                    data = get_stock_data(symbol, period=default_period)
                    if data is not None and not data.empty:
                        fig = create_stock_chart(data, symbol)
                        st.plotly_chart(fig, use_container_width=True)
                        break

# Chat interface
user_input = st.chat_input("Ask about any stock or market analysis...")

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
                # Get response from selected model
                model = available_models[selected_model]
                response = get_model_response(user_input, model)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Display the response
                st.markdown(response, unsafe_allow_html=True)
                
                # Check for stock symbols to show charts
                words = user_input.split()
                for word in words:
                    # Clean the word of punctuation
                    clean_word = word.strip('.,;:!?()')
                    if clean_word.upper() in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"] or \
                       (clean_word.isupper() and len(clean_word) <= 5 and clean_word.isalpha()):
                        symbol = clean_word.upper()
                        data = get_stock_data(symbol, period=default_period)
                        if data is not None and not data.empty:
                            fig = create_stock_chart(data, symbol)
                            st.plotly_chart(fig, use_container_width=True)
                            break
                
            except Exception as e:
                error_msg = f"Error getting response from model: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}. Please try again."})