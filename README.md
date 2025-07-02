# LSTM-stock-predict
# 📈 LSTM Stock Direction Predictor

A user-friendly **Streamlit dashboard** that uses a **PyTorch-based LSTM neural network** to predict stock price movement (up or down) based on historical price, technical indicators, and market index context (QQQ and VIX).

---

## 🚀 Features

- 🔍 **Select stock ticker** from a list of popular companies or enter a custom symbol
- 📅 **Flexible date range** selection for training and prediction
- ⚙️ **Customize model parameters**: sequence length, batch size, epochs, learning rate
- 📊 **Real-time visualizations**: price trends, correlation heatmap, prediction accuracy
- 🧠 **LSTM model training and testing** on selected dataset
- 🔮 **Next-day direction prediction** with probability

---

## 🛠️ Installation

This project is designed to run on **Google Colab** or locally with Python.

### ▶️ Run in Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com/)
2. Mount your Google Drive
3. Paste the full script or run the saved `.py` file:
   ```bash
   !streamlit run /content/drive/MyDrive/py/lstm_stock_predict.py

