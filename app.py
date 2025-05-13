import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

st.title("Stock Market Forecasting")

# User input
symbol = st.text_input("Enter Stock Symbol", "AAPL")
data = yf.download(symbol, start="2020-01-01", end="2024-01-01")
st.write("Latest Data", data.tail())

# ARIMA modeling
model = ARIMA(data['Close'], order=(5,1,0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=30)

# Plotting
st.subheader("Forecast for Next 30 Days")
fig, ax = plt.subplots()
ax.plot(data['Close'], label="Historical")
ax.plot(pd.date_range(data.index[-1], periods=30, freq='D'), forecast, label="Forecast", color='orange')
ax.legend()
st.pyplot(fig)