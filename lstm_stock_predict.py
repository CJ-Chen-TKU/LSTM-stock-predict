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

# --- Ticker Selection ---
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
selected_symbol = st.selectbox("Choose a stock:", options=list(popular_tickers.keys()),
                                format_func=lambda x: f"{x} - {popular_tickers[x]}")
manual_override = st.text_input("Or enter a custom symbol (e.g., ^GSPC for S&P 500):", "").strip().upper()
ticker = manual_override if manual_override else selected_symbol

# --- Key Inputs ---
train_end_date = st.date_input("🗓️ Training End Date", pd.to_datetime("2023-12-31"))
sequence_length = st.number_input("Sequence Length (days)", min_value=3, max_value=60, value=10)
batch_size = st.number_input("Batch Size", min_value=8, max_value=128, value=32)
epochs = st.number_input("Epochs", min_value=1, max_value=50, value=10)
learning_rate = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-1, value=0.001, format="%.5f")

# --- Computed Ranges ---
indicator_buffer = 50  # to support SMA_50
data_start_date = train_end_date - pd.Timedelta(days=sequence_length + indicator_buffer)
end_date = train_end_date + pd.Timedelta(days=30)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

@st.cache_data(show_spinner=False)
def load_data_with_indexes(symbol, start, end):
    stock = yf.download(symbol, start=start, end=end, progress=False)
    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = stock.columns.droplevel(0)

    qqq = yf.download("QQQ", start=start, end=end, progress=False)
    if isinstance(qqq.columns, pd.MultiIndex):
        qqq.columns = qqq.columns.droplevel(0)
    qqq = qqq[["Close"]].rename(columns={"Close": "QQQ_Close"})

    vix = yf.download("^VIX", start=start, end=end, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.droplevel(0)
    vix = vix[["Close"]].rename(columns={"Close": "VIX_Close"})

    df = stock.join([qqq, vix], how="outer")
    df.dropna(inplace=True)

    df["SMA_10"] = SMAIndicator(df["Close"], window=10).sma_indicator()
    df["SMA_50"] = SMAIndicator(df["Close"], window=50).sma_indicator()
    df["RSI"] = RSIIndicator(df["Close"]).rsi()
    macd = MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["QQQ_pct"] = df["QQQ_Close"].pct_change()
    df["VIX_pct"] = df["VIX_Close"].pct_change()
    df["Tomorrow_Close"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow_Close"] > df["Close"]).astype(int)
    df.dropna(inplace=True)
    return df

class StockDataset(Dataset):
    def __init__(self, data, feature_cols, target_col, seq_length):
        self.features = data[feature_cols].values
        self.targets = data[target_col].values
        self.seq_length = seq_length

    def __len__(self):
        return len(self.targets) - self.seq_length

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_length]
        y = self.targets[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out).squeeze()

def train(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            preds.extend((output.cpu().numpy() > 0.5).astype(int))
            targets.extend(y_batch.numpy().astype(int))
    acc = accuracy_score(targets, preds)
    cm = confusion_matrix(targets, preds)
    return acc, cm, preds, targets

# === Main logic ===
if st.button("🚀 Train and Predict"):
    df = load_data_with_indexes(ticker, data_start_date, end_date)
    df_train = df[df.index <= train_end_date]
    df_test = df[df.index > train_end_date]

    features = ["SMA_10", "SMA_50", "RSI", "MACD", "MACD_signal", "QQQ_pct", "VIX_pct"]
    target = "Target"

    if len(df_train) < sequence_length + 1 or len(df_test) < sequence_length + 1:
        st.error("Not enough data for training or testing.")
        st.stop()

    train_dataset = StockDataset(df_train, features, target, sequence_length)
    test_dataset = StockDataset(df_test, features, target, sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = LSTMClassifier(input_size=len(features)).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    progress_bar = st.progress(0)
    epoch_display = st.empty()

    for epoch in range(epochs):
        loss = train(model, train_loader, criterion, optimizer)
        acc, cm, preds, targets = evaluate(model, test_loader)
        epoch_display.text(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss:.4f} | Test Acc: {acc:.2%}")
        progress_bar.progress((epoch + 1) / epochs)

    st.subheader("✅ Evaluation Results")
    st.write(pd.DataFrame(cm, index=["Actual Down", "Actual Up"], columns=["Pred Down", "Pred Up"]))
    st.write(f"Test Accuracy: **{acc:.2%}**")

    pred_data = df_test.iloc[sequence_length:].copy()
    pred_data["Prediction"] = preds
    pred_data["Target"] = targets

    st.line_chart(pred_data[["Close"]])
    st.line_chart(pred_data[["Target", "Prediction"]])

    st.subheader("🔮 Next-Day Forecast")
    last_seq = torch.tensor(df_test[features].values[-sequence_length:], dtype=torch.float32).unsqueeze(0).to(device)
    pred_prob = model(last_seq).item()
    st.write(f"Probability of price increase tomorrow: **{pred_prob:.2%}**")

