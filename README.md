# Stock Market ChatBot 📊

A Streamlit-based application that combines real-time stock market data with AI-powered analysis.

## Features

- **Real-Time Stock Data**: Fetch up-to-date financial information from Alpha Vantage API
- **Interactive Charts**: View Candlestick, Line, or OHLC charts with customizable technical indicators
- **Multi-Stock Comparison**: Compare performance of up to 5 stocks simultaneously
- **AI Analysis**: Powered by Groq's large language models including:
  - Llama-3.3-70B
  - Gemma-2-9B
  - DeepSeek-R1-Distill-Llama-70B
- **Context-Aware Responses**: AI assistant analyzes loaded stock data to provide relevant insights
- **Detailed Company Information**: Access key statistics, market metrics, and company descriptions
- **Persistent Chat Experience**: Chat with the AI without triggering data reloads

## Usage

1. Select stocks from the sidebar (up to 5 for comparison)
2. Choose chart type and technical indicators
3. Customize time period and data range
4. Ask the AI assistant questions about the loaded stocks or general financial topics
5. Chat continuously without worrying about data refreshing unnecessarily

## Technical Details

- Built with Streamlit, Plotly, Pandas, and LangChain
- Alpha Vantage API for financial data (free tier: 5 calls/min, 100 calls/day)
- Groq API for AI model inference
- Comprehensive error handling and rate limiting
- Session state management for optimal user experience

## Implementation Highlights

- **Data Caching**: Optimizes API usage by storing recent stock data
- **Responsive Design**: Adapts to different screen sizes and devices
- **Advanced Error Handling**: Gracefully manages API limits and connectivity issues  
- **Real-time Data Analysis**: Calculates technical indicators and performance metrics on the fly
- **Data Visualization**: Presents complex financial data through intuitive interactive charts
- **Smart Rerendering**: Prevents unnecessary data reloading during chat interactions
- **State Management**: Tracks input changes to determine when data refresh is needed
- **Context Preservation**: Maintains data context for AI to provide relevant insights

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages: streamlit, plotly, pandas, langchain_groq, requests, python-dotenv

### Environment Setup
1. Clone this repository
2. Create a `.env` file with the following keys:
   ```
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
   GROQ_API_KEY=your_groq_api_key
   ```
3. Install requirements:
   ```
   pip install -r requirements.txt
   ```
4. Run the application:
   ```
   streamlit run alpha_vantage.py
   ```

### API Keys
- Get your Alpha Vantage API key from: https://www.alphavantage.co/support/#api-key
- Get your Groq API key from: https://console.groq.com/

## Performance Optimization Tips
- Be mindful of Alpha Vantage API limits (5 calls/min, 100 calls/day on free tier)
- Use compact data mode when working with multiple stocks
- Allow sufficient time between stock switches to respect API rate limits

## Note

This application is for informational purposes only and does not constitute financial advice. Financial decisions should be made after consulting qualified professionals.