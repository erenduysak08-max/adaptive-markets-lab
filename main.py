import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
data = yf.download('SPY', start='2020-01-01', end='2021-01-01', auto_adjust = False, multi_level_index = False)
print(data.head())
print(data.tail())
print(data.shape)
print(data.columns.isna().sum())
print(data.columns)
prices = data["Adj Close"]
returns = (prices - prices.shift(-1))/prices
print(returns.head(10))
print(prices.pct_change().head(10))
log_returns = np.log(prices/prices.shift(-1))
print(log_returns.head(10))

plt.plot(prices, "b-")
plt.plot(returns, "r-")
plt.show()