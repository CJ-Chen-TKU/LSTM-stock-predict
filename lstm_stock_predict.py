 # py code beginning

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="LSTM Stock Predictor", layout="wide")
st.title("📈 Stock Direction Prediction with LSTM and Market Index Context")

if "loaded_data" not in st.session_state:
    st.session_state.loaded_data = None

# --- Ticker Selection ---
st.subheader("🔎 Select a Stock Ticker")

popular_tickers = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com, Inc.",
    "TSLA": "Tesla Inc.",
    "META": "Meta Platforms, Inc.",
    "NVDA": "NVIDIA Corporation",
    "NFLX": "Netflix, Inc.",
    "BRK-B": "Berkshire Hathaway Inc.",
    "JPM": "JPMorgan Chase & Co.",
}

selected_symbol = st.selectbox(
    "Choose a stock:", options=list(popular_tickers.keys()),
    format_func=lambda x: f"{x} - {popular_tickers[x]}"
)
manual_override = st.text_input("Or enter a custom symbol (e.g., ^GSPC for S&P 500):", "").strip().upper()
ticker = manual_override if manual_override else selected_symbol

# --- Date & Hyper‑parameters ---
start_date = st.date_input("Data Start Date", pd.to_datetime("2018-01-01"))
end_date = st.date_input("Data End Date", pd.to_datetime("2024-06-30"))
train_start_date = st.date_input("Train Start Date", pd.to_datetime("2018-01-01"))
train_end_date = st.date_input("Train End Date", pd.to_datetime("2023-12-31"))

sequence_length = st.number_input("Sequence Length (trading days)", 3, 60, 10)
batch_size     = st.number_input("Batch Size", 8, 128, 32)
epochs         = st.number_input("Epochs", 1, 50, 10)
learning_rate  = st.number_input("Learning Rate", 1e-5, 1e-1, 0.001, format="%.5f")

forecast_target_date = st.date_input("Forecast Target Date", value=pd.to_datetime(train_end_date) + pd.Timedelta(days=1))
forecast_target_date = pd.to_datetime(forecast_target_date)

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

# ------------------------------------------------------------------
#  Data loading helper
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data_with_indexes(symbol, start, end):
    stock = yf.download(symbol, start=start, end=end, progress=False)
    qqq   = yf.download("QQQ",   start=start, end=end, progress=False)[["Close"]].rename(columns={"Close": "QQQ_Close"})
    vix   = yf.download("^VIX",  start=start, end=end, progress=False)[["Close"]].rename(columns={"Close": "VIX_Close"})

    df = stock.join([qqq, vix], how="outer").dropna()
    if df.empty:
        return df

    # Technical indicators
    df["SMA_10"]   = SMAIndicator(df["Close"], window=10).sma_indicator()
    df["SMA_50"]   = SMAIndicator(df["Close"], window=50).sma_indicator()
    df["RSI"]      = RSIIndicator(df["Close"]).rsi()
    macd           = MACD(df["Close"])
    df["MACD"]         = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    # Market context
    df["QQQ_pct"]  = df["QQQ_Close"].pct_change()
    df["VIX_pct"]  = df["VIX_Close"].pct_change()
    # Target label
    df["Tomorrow_Close"] = df["Close"].shift(-1)
    df["Target"]         = (df["Tomorrow_Close"] > df["Close"]).astype(int)

    return df.dropna()

# ------------------------------------------------------------------
#  Dataset & model
# ------------------------------------------------------------------
class StockDataset(Dataset):
    def __init__(self, data, feature_cols, target_col, seq_length):
        self.features = data[feature_cols].values
        self.targets  = data[target_col].values
        self.seq_len  = seq_length

    def __len__(self):
        return len(self.targets) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = out[:, -1, :]
        return self.sig(self.fc(out)).squeeze()

# ------------------------------------------------------------------
#  Utility functions
# ------------------------------------------------------------------

def train_epoch(model, loader, crit, opt):
    model.train(); total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(); out = model(xb); loss = crit(out, yb)
        loss.backward(); opt.step(); total += loss.item()*xb.size(0)
    return total/len(loader.dataset)

def evaluate(model, loader):
    model.eval(); pred, true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device)); pred.extend((out.cpu()>0.5).int().numpy()); true.extend(yb.numpy())
    acc = accuracy_score(true, pred); cm = confusion_matrix(true, pred)
    return acc, cm, pred, true

# ------------------------------------------------------------------
#  Buttons & interactions
# ------------------------------------------------------------------
if st.button("📊 Load & Show Price Chart"):
    data = load_data_with_indexes(ticker, start_date, end_date)
    if data.empty:
        st.error("No data returned for these dates.")
    else:
        st.session_state.loaded_data = data
        st.line_chart(data[["Close", "QQQ_Close", "VIX_Close"]])

if st.button("📈 Show Correlation Matrix") and st.session_state.loaded_data is not None:
    corr = st.session_state.loaded_data[["Close", "SMA_10", "SMA_50", "RSI", "MACD", "MACD_signal", "QQQ_pct", "VIX_pct"]].corr()
    fig, ax = plt.subplots(); sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax); st.pyplot(fig)
elif st.button("📈 Show Correlation Matrix"):
    st.warning("Load data first.")

# -------------------- TRAIN & PREDICT -----------------------------
if st.button("🚀 Train and Predict"):
    # Make sure forecast date is within data end range
    if forecast_target_date > end_date:
        end_for_download = forecast_target_date + pd.Timedelta(days=1)
    else:
        end_for_download = end_date

    data = load_data_with_indexes(ticker, start_date, end_for_download)
    if data.empty:
        st.error("No data found. Please adjust dates or ticker.")
        st.stop()

    features = ["SMA_10", "SMA_50", "RSI", "MACD", "MACD_signal", "QQQ_pct", "VIX_pct"]
    target_col = "Target"

    train_df = data[(data.index >= pd.to_datetime(train_start_date)) & (data.index <= pd.to_datetime(train_end_date))]
    test_df  = data[data.index > pd.to_datetime(train_end_date)]

    if len(train_df) < sequence_length+1 or len(test_df) < sequence_length+1:
        st.error("Not enough data in chosen windows for training/testing.")
        st.stop()

    train_ds = StockDataset(train_df, features, target_col, sequence_length)
    test_ds  = StockDataset(test_df,  features, target_col, sequence_length)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_ld  = DataLoader(test

