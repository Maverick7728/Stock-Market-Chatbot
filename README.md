# Stock Market Analysis Chatbot

This is an AI-powered chatbot that helps you analyze stocks and answer questions about the market. It uses OpenAI's language model and real-time stock data from Yahoo Finance.

## Features

- Real-time stock data analysis
- Interactive stock charts
- Natural language processing for market analysis
- User-friendly chat interface
- Support for multiple stock symbols

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run stock_chatbot.py
   ```
2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)
3. Start asking questions about stocks!

## Example Questions

- What's the current price of AAPL?
- Show me the performance of MSFT over the last month
- What's the market sentiment for GOOGL?
- Compare the performance of AAPL and MSFT

## Note

Make sure you have a valid OpenAI API key and a stable internet connection to access real-time stock data. 