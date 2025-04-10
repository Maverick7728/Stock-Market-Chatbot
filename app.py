import streamlit as st
import finnhub
import datetime
import plotly.graph_objects as go
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Finnhub client
finnhub_api_key = os.getenv("FINNHUB_API_KEY")
finnhub_client = finnhub.Client(api_key=finnhub_api_key)

# Initialize LLM Models
def initialize_models():
    models = {}
    groq_api_key = os.getenv("GROQ_API_KEY")
    try:
        models["Llama-3.3-70B"] = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=None,
            api_key=groq_api_key
        )
        models["Gemma-2-9B"] = ChatGroq(
            model="gemma2-9b-it",
            temperature=0.7,
            max_tokens=None,
            api_key=groq_api_key
        )
    except Exception as e:
        st.error(f"⚠️ Error initializing models: {str(e)}")
        return {}
    return models

# Keyword extraction
def extract_finance_keywords(text):
    stock_map = {
        "APPLE": "AAPL", "MICROSOFT": "MSFT", "GOOGLE": "GOOGL", 
        "AMAZON": "AMZN", "META": "META", "TESLA": "TSLA", "NVIDIA": "NVDA",
        "NETFLIX": "NFLX", "BERKSHIRE": "BRK.B", "JPMORGAN": "JPM",
        "WALMART": "WMT", "DISNEY": "DIS", "COCA-COLA": "KO"
    }
    words = text.upper().split()
    keywords, symbols = [], []
    finance_terms = ["STOCK", "MARKET", "NASDAQ", "DOW", "S&P", "ETF"]
    for word in [w.strip('.,;:!?()[]{}"\'-') for w in words]:
        if word.isalpha() and 1 <= len(word) <= 5 and word.isupper():
            symbols.append(word)
        if word in finance_terms and word not in keywords:
            keywords.append(word)
    for company, symbol in stock_map.items():
        if company in text.upper() and symbol not in symbols:
            symbols.append(symbol)
    return {"keywords": keywords, "symbols": symbols}

# Fetch stock data using Finnhub
def fetch_stock_data(symbols):
    results, charts = [], {}
    for symbol in symbols[:5]:
        try:
            quote = finnhub_client.quote(symbol)
            profile = finnhub_client.company_profile2(symbol=symbol)
            if not quote or quote['c'] == 0:
                results.append(f"No data available for {symbol}")
                continue
            current = quote['c']
            prev = quote['pc']
            change = current - prev
            percent = (change / prev) * 100 if prev != 0 else 0
            direction = "🔼" if change >= 0 else "🔽"
            name = profile.get('name', symbol)
            charts[symbol] = {"symbol": symbol, "name": name}
            market_cap = profile.get('marketCapitalization', 'N/A')
            if market_cap != 'N/A':
                market_cap = f"${market_cap:.2f}B"
            results.append(f"""**{name} ({symbol})**: ${current:.2f} {direction} {change:.2f} ({percent:.2f}%)  
**Market Cap**: {market_cap}  
**Data as of**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
""")
        except Exception as e:
            results.append(f"Finnhub error: {str(e)}")
    return {"text": "\n".join(results), "charts": charts}

# Generate chart from Finnhub historical data
def generate_stock_chart(symbol, resolution="D", count=30):
    try:
        now = int(datetime.datetime.now().timestamp())
        past = now - count * 24 * 60 * 60
        candles = finnhub_client.stock_candles(symbol, resolution, past, now)
        if candles['s'] != 'ok':
            return None
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=[datetime.datetime.fromtimestamp(ts) for ts in candles['t']],
            open=candles['o'], high=candles['h'],
            low=candles['l'], close=candles['c'],
            name=symbol
        ))
        fig.update_layout(title=f"{symbol} - Price Chart", template="plotly_dark", height=500)
        return fig
    except Exception as e:
        st.error(f"Chart error for {symbol}: {str(e)}")
        return None

# Get model response
def get_model_response(question, model):
    finance_data = extract_finance_keywords(question)
    keywords, symbols = finance_data["keywords"], finance_data["symbols"]
    finance_context, charts_to_display = "", {}
    if symbols:
        stock_data = fetch_stock_data(symbols)
        finance_context += "REAL-TIME STOCK DATA:\n\n" + stock_data["text"] + "\n\n"
        charts_to_display.update(stock_data["charts"])
    prompt = f"""You are a financial advisor chatbot.  
USER QUESTION: {question}  
{finance_context}  
Your task:  
1. Analyze the user's question  
2. Use real-time finance data  
3. Be concise and factual  
4. Add disclaimers for financial advice  
"""
    try:
        response = model.invoke(prompt)
        return {"content": response.content, "charts": charts_to_display}
    except Exception as e:
        return {"content": f"Error: {str(e)}", "charts": {}}
    


# --- Streamlit UI ---
st.set_page_config(page_title="Stock Market Assistant", page_icon="📈", layout="wide")

st.markdown("""
<style>
.main, .stApp {background-color: #1e1e1e; color: #e0e0e0;}
.stChatMessage {background-color: #2d2d2d; border-radius: 10px; padding: 10px; margin: 10px 0;}
.stButton>button {background-color: #1f77b4; color: white; border-radius: 20px;}
.stock-card {background-color: #2d2d2d; border-radius: 10px; padding: 20px; margin: 10px 0;}
h1, h2, h3, h4, h5, h6 {color: #e0e0e0;}
</style>
""", unsafe_allow_html=True)

available_models = initialize_models()

# Sidebar
with st.sidebar:
    st.title("📊 Stock Market Chatbot")
    if available_models:
        selected_model = st.selectbox("Choose the AI model:", list(available_models.keys()))
    else:
        selected_model = None
    if st.button("Test Data Connection"):
        try:
            quote = finnhub_client.quote("AAPL")
            st.success(f"✅ AAPL price: ${quote['c']:.2f}")
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")
    st.markdown("### Features")
    st.markdown("""
    - Stock market analysis with real-time data
    - Interactive stock price charts
    - Market trend visualization
    - Company insights
    - Financial guidance
    """)
    
    st.markdown("### Sample Questions")
    st.markdown("""
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

question = st.text_input("💬 Ask your question:")
if question and selected_model:
    with st.spinner("Analyzing your question..."):
        response_data = get_model_response(question, available_models[selected_model])
    st.markdown("#### 💡 AI Response")
    st.markdown(response_data["content"])
    for symbol in response_data["charts"]:
        fig = generate_stock_chart(symbol)
        if fig: st.plotly_chart(fig, use_container_width=True)
