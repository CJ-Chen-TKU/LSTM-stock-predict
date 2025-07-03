  # py code beginning

import yfinance as yf
import pandas as pd
import streamlit as st
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
import torch
import torch.nn as nn
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

# --- App Config ---
st.set_page_config(page_title="LSTM Stock Predictor", layout="wide")
st.title("📈 Stock Direction Prediction with LSTM and Market Index Context")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.write(f"Using device: {device}")

# --- Sidebar: Inputs ---
st.sidebar.header("1️⃣ Select Stock")
popular_tickers = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.", "AMZN": "Amazon.com, Inc.",
    "TSLA": "Tesla Inc.",  "META": "Meta Platforms, Inc.",
    "NVDA": "NVIDIA Corporation", "NFLX": "Netflix, Inc.",
    "BRK-B": "Berkshire Hathaway Inc.", "JPM": "JPMorgan Chase & Co."
}
selected_symbol = st.sidebar.selectbox(
    "Choose a stock:", options=list(popular_tickers.keys()),
    format_func=lambda x: f"{x} - {popular_tickers[x]}"
)
manual_override = st.sidebar.text_input("Or enter a custom symbol (e.g., ^GSPC):", "").strip().upper()
ticker = manual_override if manual_override else selected_symbol

st.sidebar.header("2️⃣ Set Dates & Hyperparameters")
today = pd.Timestamp.today().normalize()
yesterday = today - pd.Timedelta(days=1)

# Date inputs return datetime.date; convert to pd.Timestamp immediately for consistency
def to_timestamp(date_obj):
    return pd.Timestamp(date_obj)

start_date = to_timestamp(st.sidebar.date_input("Data Start Date", pd.to_datetime("2018-01-01")))
end_date = to_timestamp(st.sidebar.date_input("Data End Date", yesterday))
train_start_date = to_timestamp(st.sidebar.date_input("Train Start Date", pd.to_datetime("2018-01-01")))
train_end_date = to_timestamp(st.sidebar.date_input("Train End Date", pd.to_datetime("2023-12-31")))
forecast_target_date = to_timestamp(st.sidebar.date_input("Forecast Target Date", today))

st.sidebar.markdown(
    """
    **📅 Date Selection Reminder**
    Please ensure:
    `start_date < train_start_date < train_end_date < forecast_target_date <= end_date`
    """
)

sequence_length = st.sidebar.number_input("Sequence Length (days)", 3, 60, 10)
batch_size = st.sidebar.number_input("Batch Size", 8, 128, 32)
epochs = st.sidebar.number_input("Epochs", 1, 50, 10)
learning_rate = st.sidebar.number_input("Learning Rate", 1e-5, 1e-1, 0.001, format="%.5f")

# --- Data Loading Function with Debug ---
@st.cache_data(show_spinner=False)
def load_data_with_indexes(symbol, start, end):
    try:
        start = pd.to_datetime(start)
        end = min(pd.to_datetime(end), pd.Timestamp.today())
        today = pd.Timestamp.today().normalize()

        # ✅ Validate ticker
        if not yf.Ticker(symbol).info.get("regularMarketPrice"):
            st.error(f"⚠️ '{symbol}' is not a valid ticker.")
            return pd.DataFrame()

        st.write(f"🔍 Downloading {symbol} from {start.date()} to {end.date()}...")
        stock = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
###
        if stock.empty:
            st.error("❗ Stock data is empty.")
            return pd.DataFrame()

        st.write("📄 Raw stock data preview:")
        st.dataframe(stock.head())

        # ✅ Flatten MultiIndex columns if needed
        if isinstance(stock.columns, pd.MultiIndex):
            stock.columns = stock.columns.map(lambda x: f"{x[0]}_{x[1]}")
        else:
            stock.columns = [f"{col}_{symbol}" for col in stock.columns]

        # ✅ Rename Close_{symbol} → Close for consistency
        close_col = f"Close_{symbol}"
        if close_col not in stock.columns:
            st.error(f"❗ Expected column '{close_col}' not found.")
            return pd.DataFrame()
        stock.rename(columns={close_col: "Close"}, inplace=True)

        # ✅ QQQ download and renaming
        st.write("📥 Downloading QQQ and VIX...")
        qqq = yf.download("QQQ", start=start, end=end, progress=False, auto_adjust=True)
        
        if qqq.empty:
            st.error("❗ QQQ data is empty.")
            return pd.DataFrame()

        # 🔧 Flatten QQQ columns if needed
        if isinstance(qqq.columns, pd.MultiIndex):
            qqq.columns = qqq.columns.map(lambda x: f"{x[0]}_{x[1]}")
        else:
            qqq.columns = [f"{col}_QQQ" for col in qqq.columns]

        # ✅ Rename Close column
        close_col_qqq = [col for col in qqq.columns if "Close" in col]
        if not close_col_qqq:
            st.error("❗ QQQ data missing 'Close' column.")
            return pd.DataFrame()
        qqq.rename(columns={close_col_qqq[0]: "QQQ_Close"}, inplace=True)
        

        # ✅ VIX download and renaming
        vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
        if vix.empty:
            st.error("❗ VIX data is empty.")
            return pd.DataFrame()

        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.map(lambda x: f"{x[0]}_{x[1]}")
        else:
            vix.columns = [f"{col}_VIX" for col in vix.columns]

        close_col_vix = [col for col in vix.columns if "Close" in col]
        if not close_col_vix:
            st.error("❗ VIX data missing 'Close' column.")
            return pd.DataFrame()
        vix.rename(columns={close_col_vix[0]: "VIX_Close"}, inplace=True)
        
        # ✅ Join stock, QQQ, and VIX
        df = stock.join([qqq, vix], how="outer")
        st.write(f"🔗 Data shape after join: {df.shape}")
        df.dropna(inplace=True)
        st.write(f"🧹 Data shape after dropna: {df.shape}")

        # ✅ Technical indicators
        df["SMA_10"] = SMAIndicator(df["Close"], window=10).sma_indicator()
        df["SMA_50"] = SMAIndicator(df["Close"], window=50).sma_indicator()
        df["RSI"] = RSIIndicator(df["Close"]).rsi()
        macd = MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()

        # ✅ Targets and percent changes
        df["QQQ_pct"] = df["QQQ_Close"].pct_change()
        df["VIX_pct"] = df["VIX_Close"].pct_change()
        df["Tomorrow_Close"] = df["Close"].shift(-1)
        df["Target"] = (df["Tomorrow_Close"] > df["Close"]).astype(int)

        df.dropna(inplace=True)
        st.write("✅ Final columns:")
        st.write(df.columns.tolist())

        return df

    except Exception as e:
        st.error(f"❗ Error loading data: {e}")
        return pd.DataFrame()




# --- UI: Layout ---
left, right = st.columns([1, 2])

with left:
    if st.button("📊 Load & Show Price Chart"):
        safe_end = min(end_date + pd.Timedelta(days=1), pd.Timestamp.today())
        df = load_data_with_indexes(ticker, start_date, safe_end)
        if df.empty:
            st.error("Data is empty or failed to load.")
        else:
            st.session_state.loaded_data = df
            st.success(f"Successfully loaded {len(df)} rows")

with right:
    if "loaded_data" in st.session_state:
        st.subheader(f"{ticker} Price & Index Chart")
        st.line_chart(st.session_state.loaded_data[["Close", "QQQ_Close", "VIX_Close"]])

# --- Training and Prediction ---
if st.button("🚀 Train and Predict"):
    st.write(f"Validating date order...")

    # Ensure comparisons use pd.Timestamp for consistency
    if not (start_date < train_start_date < train_end_date < forecast_target_date <= end_date):
        st.warning(
            f"❗ Invalid date order. start_date < train_start_date < train_end_date < forecast_target_date <= end_date. "
            f"Your input: start_date={start_date.date()}, train_start_date={train_start_date.date()}, "
            f"train_end_date={train_end_date.date()}, forecast_target_date={forecast_target_date.date()}, "
            f"end_date={end_date.date()}"
        )
        st.stop()

    # Download data including forecast target date +1 for training/testing
    safe_end_for_download = min(end_date, forecast_target_date + pd.Timedelta(days=1), pd.Timestamp.today())
    st.write(f"Loading data from {start_date.date()} to {safe_end_for_download.date()} for training and testing...")
    data = load_data_with_indexes(ticker, start_date, safe_end_for_download)
    if data.empty:
        st.error("❗ No data returned or data is empty.")
        st.stop()


# Filter data by date range
    features = ["SMA_10", "SMA_50", "RSI", "MACD", "MACD_signal", "QQQ_pct", "VIX_pct"]

    target_col = "Target"
    train_df = data[(data.index >= train_start_date) & (data.index <= train_end_date)]
    test_df = data[data.index > train_end_date]

    st.write(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")

    if len(train_df) < sequence_length + 1 or len(test_df) < sequence_length + 1:
        st.warning("❗ Not enough data for training or testing. Please adjust date ranges.")
        st.stop()

    # Prepare Dataset and DataLoader
    train_ds = StockDataset(train_df, features, target_col, sequence_length)
    test_ds = StockDataset(test_df, features, target_col, sequence_length)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_ld = DataLoader(test_ds, batch_size=batch_size)

    # Initialize model, loss function, and optimizer
    model = LSTMClassifier(input_size=len(features)).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    progress = st.progress(0)
    status = st.empty()
    for ep in range(epochs):
        loss = train_epoch(model, train_ld, criterion, optimizer)
        acc, cm, _, _ = evaluate(model, test_ld)
        status.text(f"Epoch {ep+1}/{epochs} | Loss: {loss:.4f} | Test Acc: {acc:.2%}")
        progress.progress((ep + 1) / epochs)

    st.subheader("✅ Confusion Matrix")
    st.write(pd.DataFrame(cm, index=["Actual Down", "Actual Up"], columns=["Pred Down", "Pred Up"]))
    st.write(f"Test Accuracy: **{acc:.2%}**")

    # Prediction Results
    test_preds, test_true = [], []
    with torch.no_grad():
        for xb, yb in test_ld:
            out = model(xb.to(device))
            test_preds.extend(out.cpu().numpy())
            test_true.extend(yb.numpy())

    idx_offset = sequence_length
    pred_df = test_df.iloc[idx_offset:].copy()
    pred_df["Pred_Prob"] = test_preds
    pred_df["Pred_Label"] = (pred_df["Pred_Prob"] > 0.5).astype(int)

    st.subheader("📉 Prediction vs Close Price")
    st.line_chart(pred_df[["Close"]])
    st.line_chart(pred_df[["Target", "Pred_Label"]])

    # Forecast for specific target date
    if forecast_target_date not in data.index:
        st.warning("⚠️ Forecast target date is not in the downloaded data. Please check your selected dates.")
    else:
        seq_end_idx = data.index.get_loc(forecast_target_date) - 1
        seq_start_idx = seq_end_idx - sequence_length + 1
        if seq_start_idx < 0:
            st.warning("⚠️ Not enough prior data for the selected sequence length.")
        else:
            last_seq = torch.tensor(
                data.iloc[seq_start_idx:seq_end_idx + 1][features].values,
                dtype=torch.float32
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                prob_up = model(last_seq).item()
            st.subheader("🔮 Forecast Result")
            st.write(f"Probability that **{ticker}** increases on {forecast_target_date.date()}: **{prob_up:.2%}**")

