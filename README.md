# LSTM-stock-predict
# 📈 This is an interactive Streamlit web app that downloads daily stock price data from Alpha Vantage, computes technical indicators, and trains a Long Short-Term Memory (LSTM) neural network to predict next-day stock price movement direction. The app also supports multi-day forecast simulation with visualized price trends.
---

## 🚀 Features
Download daily OHLCV stock data via the Alpha Vantage API (free tier)
Compute technical indicators: SMA (10 & 50 days), RSI, MACD
Train an LSTM-based binary classifier (up/down) using PyTorch
Predict next-day price direction and simulate multi-day forecasts with predicted trend price changes
Interactive UI to select stock ticker, date ranges, and model hyperparameters
Visualization with Plotly candlestick charts and prediction results
Model training and data caching for efficient usage

## Usage
1. Clone the repository
2. Create and activate a virtual environment (optional but recommended)
3. Install dependencies
4. Set your Alpha Vantage API key
5. Run the Streamlit app
6. Interact with the app
    Select a stock from popular tickers or enter a custom symbol.
    Pick data start/end dates, training period, and forecast date.
    Adjust model hyperparameters like sequence length, batch size, epochs, and learning rate.
    Click "Load & Show Price Chart" to download and visualize stock data.
    Click "Train and Predict" to train the LSTM model and evaluate test accuracy.
    Use "Multi-Day Forecast" to simulate future price movement trends and predicted directions.

## Requirements
Python 3.8+
streamlit
pandas
numpy
torch (PyTorch)
scikit-learn
ta (Technical Analysis library)
plotly
requests

## Notes
Alpha Vantage free tier limits API calls to 5 per minute and 500 per day; the app waits 15 seconds after each data download to respect this.
The multi-day forecast simulates price movement by applying a fixed 1% up/down change based on predicted direction.
Model training requires enough historical data relative to the chosen sequence length and date ranges.
The app uses GPU acceleration if available.
Weekends are skipped in forecast simulation to reflect trading days.

## File Structure
lstm_stock_predict.py: Main Streamlit app script with all the data loading, model training, prediction, and UI logic.
requirements.txt: Required Python packages.
README.md: This documentation file.

## License
This project is licensed under the MIT License.
