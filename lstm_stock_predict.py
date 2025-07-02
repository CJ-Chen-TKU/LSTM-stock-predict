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

# ---------------------- Streamlit page config --------------------------
st.set_page_config(page_title="LSTM Stock Predictor", layout="wide")
st.title("📈 Stock Direction Prediction with LSTM and Market Index Context")

# ---------------------- Session‑state init -----------------------------
if "loaded_data" not in st.session_state:
    st.session_state.loaded_data = None

# ---------------------- Ticker selection -------------------------------
st.subheader("🔎 Select a Stock Ticker")
popular_tickers = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.", "AMZN": "Amazon.com, Inc.",
    "TSLA": "Tesla Inc.",  "META": "Meta Platforms, Inc.",
    "NVDA": "NVIDIA Corporation", "NFLX": "Netflix, Inc.",
    "BRK-B": "Berkshire Hathaway Inc.", "JPM": "JPMorgan Chase & Co.",
}
selected_symbol = st.selectbox(
    "Choose a stock:", options=list(popular_tickers.keys()),
    format_func=lambda x: f"{x} - {popular_tickers[x]}"
)
manual_override = st.text_input("Or enter a custom symbol (e.g., ^GSPC for S&P 500):", "").strip().upper()
ticker = manual_override if manual_override else selected_symbol

# ---------------------- Date & hyper‑parameters ------------------------
st.subheader("⚙️ Date Range & Hyper‑parameters")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Data Start Date", pd.to_datetime("2018-01-01"))
    train_start_date = st.date_input("Train Start Date", pd.to_datetime("2018-01-01"))
    sequence_length = st.number_input("Sequence Length (trading days)", 3, 60, 10)
    batch_size     = st.number_input("Batch Size", 8, 128, 32)
with col2:
    end_date   = st.date_input("Data End Date",   pd.to_datetime("2024-06-30"))
    train_end_date = st.date_input("Train End Date", pd.to_datetime("2023-12-31"))
    epochs         = st.number_input("Epochs", 1, 50, 10)
    learning_rate  = st.number_input("Learning Rate", 1e-5, 1e-1, 0.001, format="%.5f")

forecast_target_date = st.date_input(
    "Forecast Target Date",
    value=pd.to_datetime(train_end_date) + pd.Timedelta(days=1)
)
forecast_target_date = pd.to_datetime(forecast_target_date)

# ---------------------- Device ----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

# ---------------------- Data loader -----------------------------------
@st.cache_data(show_spinner=False)
def load_data_with_indexes(symbol, start, end):
    stock = yf.download(symbol, start=start, end=end, progress=False)

    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = stock.columns.get_level_values(0)  # 抽出欄位名稱層級 0 (Close, Open...)

    if "Close" not in stock.columns:
        st.error(f"No 'Close' column found for {symbol}. Columns: {stock.columns}")
        return pd.DataFrame()

    # 下面是 QQQ 和 VIX 的資料下載，依樣處理 MultiIndex
    qqq = yf.download("QQQ", start=start, end=end, progress=False)[["Close"]]
    if isinstance(qqq.columns, pd.MultiIndex):
        qqq.columns = qqq.columns.get_level_values(0)
    qqq.rename(columns={"Close": "QQQ_Close"}, inplace=True)

    vix = yf.download("^VIX", start=start, end=end, progress=False)[["Close"]]
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix.rename(columns={"Close": "VIX_Close"}, inplace=True)

    df = stock.join([qqq, vix], how="outer").dropna()

    # 技術指標計算
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

    return df.dropna()

    st.write(f"After dropna shape: {df.shape}")

    if df.empty:
        st.warning("DataFrame is empty after join and dropna.")
        return pd.DataFrame()

    # 之後技術指標計算
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

    df = df.dropna()
    st.write(f"After indicators and dropna shape: {df.shape}")

    return df



# ---------------------- Dataset & model -------------------------------
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

# ---------------------- Helper: train & evaluate ----------------------
def train_epoch(model, loader, crit, opt):
    model.train(); total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(); out = model(xb); loss = crit(out, yb)
        loss.backward(); opt.step()
        total += loss.item()*xb.size(0)
    return total/len(loader.dataset)

def evaluate(model, loader):
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device))
            preds.extend((out.cpu()>0.5).int().numpy())
            trues.extend(yb.numpy())
    acc = accuracy_score(trues, preds)
    cm  = confusion_matrix(trues, preds)
    return acc, cm, preds, trues

# ---------------------- UI Buttons ------------------------------------
st.divider()
colA, colB, colC = st.columns(3)

with colA:
    if st.button("📊 Load & Show Price Chart"):
        data = load_data_with_indexes(ticker, start_date, end_date)
        if data.empty:
            st.error("No data returned for these dates.")
        else:
            st.session_state.loaded_data = data
            st.line_chart(data[["Close", "QQQ_Close", "VIX_Close"]])

with colB:
    if st.button("📈 Show Correlation Matrix"):
        if st.session_state.loaded_data is not None:
            corr_cols = ["Close", "SMA_10", "SMA_50", "RSI",
                         "MACD", "MACD_signal", "QQQ_pct", "VIX_pct"]
            corr = st.session_state.loaded_data[corr_cols].corr()
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("▶  Load data first using the left button.")

with colC:
    run_train = st.button("🚀 Train and Predict")

# ---------------------- Main: Train & Predict -------------------------
if run_train:
    # Ensure forecast date is covered
    

    end_for_download = max(pd.Timestamp(end_date), pd.Timestamp(forecast_target_date) + pd.Timedelta(days=1))

    data = load_data_with_indexes(ticker, start_date, end_for_download)
    if data.empty:
        st.error("No data found. Please adjust dates or ticker.")
        st.stop()

    features = ["SMA_10", "SMA_50", "RSI", "MACD",
                "MACD_signal", "QQQ_pct", "VIX_pct"]
    target_col = "Target"

    train_df = data[(data.index >= pd.to_datetime(train_start_date)) &
                    (data.index <= pd.to_datetime(train_end_date))]
    test_df  = data[data.index > pd.to_datetime(train_end_date)]

    if len(train_df) < sequence_length+1 or len(test_df) < sequence_length+1:
        st.error("Not enough data in chosen windows for training/testing.")
        st.stop()

    train_ds = StockDataset(train_df, features, target_col, sequence_length)
    test_ds  = StockDataset(test_df,  features, target_col, sequence_length)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_ld  = DataLoader(test_ds,  batch_size=batch_size)

    model = LSTMClassifier(input_size=len(features)).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    progress = st.progress(0)
    status   = st.empty()

    for ep in range(epochs):
        loss = train_epoch(model, train_ld, criterion, optimizer)
        acc, cm, _, _ = evaluate(model, test_ld)
        status.text(f"Epoch {ep+1}/{epochs}  |  Train Loss: {loss:.4f}  |  Test Acc: {acc:.2%}")
        progress.progress((ep+1)/epochs)

    # ----------- Evaluation output ------------------------------------
    st.subheader("✅ Evaluation Results")
    st.write(pd.DataFrame(cm, index=["Actual Down", "Actual Up"],
                             columns=["Pred Down", "Pred Up"]))
    st.write(f"Test Accuracy: **{acc:.2%}**")

    # ----------- Test set predictions line chart ----------------------
    test_preds = []
    test_true  = []
    with torch.no_grad():
        for xb, yb in test_ld:
            out = model(xb.to(device))
            test_preds.extend(out.cpu().numpy())
            test_true.extend(yb.numpy())
    idx_offset = sequence_length  # align with original df index
    pred_df = test_df.iloc[idx_offset:].copy()
    pred_df["Pred_Prob"] = test_preds
    pred_df["Pred_Label"] = (pred_df["Pred_Prob"] > 0.5).astype(int)

    st.subheader("📉 Prediction vs. Close Price (Test window)")
    st.line_chart(pred_df[["Close"]])
    st.line_chart(pred_df[["Target", "Pred_Label"]])

    # ----------- Forecast for chosen target date ----------------------
    if forecast_target_date not in data.index:
        st.warning("Forecast target date is outside downloaded data range.")
    else:
        seq_end_idx = data.index.get_loc(forecast_target_date) - 1
        seq_start_idx = seq_end_idx - sequence_length + 1
        if seq_start_idx < 0:
            st.warning("Not enough prior data for chosen sequence length.")
        else:
            last_seq = torch.tensor(
                data.iloc[seq_start_idx:seq_end_idx+1][features].values,
                dtype=torch.float32
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                prob_up = model(last_seq).item()
            st.subheader("🔮 Forecast Result")
            st.write(f"Probability that **{ticker}** increases on "
                     f"{forecast_target_date.date()}: **{prob_up:.2%}**")

