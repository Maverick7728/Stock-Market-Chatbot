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

## Usage

1. Select stocks from the sidebar (up to 5 for comparison)
2. Choose chart type and technical indicators
3. Customize time period and data range
4. Ask the AI assistant questions about the loaded stocks or general financial topics

## Technical Details

- Built with Streamlit, Plotly, Pandas, and LangChain
- Alpha Vantage API for financial data (free tier: 5 calls/min, 100 calls/day)
- Groq API for AI model inference
- Comprehensive error handling and rate limiting

## Implementation Highlights

- **Data Caching**: Optimizes API usage by storing recent stock data
- **Responsive Design**: Adapts to different screen sizes and devices
- **Advanced Error Handling**: Gracefully manages API limits and connectivity issues  
- **Real-time Data Analysis**: Calculates technical indicators and performance metrics on the fly
- **Data Visualization**: Presents complex financial data through intuitive interactive charts

## Getting Started

### Prerequisites
- Python 3.8+
- Streamlit
- API keys for Alpha Vantage and Groq


## Note

This application is for informational purposes only. Financial decisions should be made after consulting qualified professionals.