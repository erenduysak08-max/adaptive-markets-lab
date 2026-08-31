import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

data = yf.download('SPY', start='2020-01-01', end='2021-01-01', auto_adjust = False, multi_level_index = False)
prices = data["Adj Close"]
returns = prices.pct_change().dropna()
signal = returns.ewm(halflife=20).mean()
position = (signal > 0).astype(int)
position_change = position.diff()

strategy_returns = position.shift(1) * returns
biased_strategy_returns = position * returns #This is the biased strategy that uses the a look ahead bias by using the current signal to determine the position for the current day.

strategy_growth = (1 + strategy_returns.fillna(0)).cumprod()
benchmark_growth = (1 + returns.fillna(0)).cumprod()
biased_growth = (1 + biased_strategy_returns.fillna(0)).cumprod()

strategy_total_return = strategy_growth.iloc[-1] - 1
benchmark_total_return = benchmark_growth.iloc[-1] - 1
biased_total_return = biased_growth.iloc[-1] - 1

backtest = pd.DataFrame({
    "Price": prices,
    "Return": returns,
    "Signal": signal,
    "Position": position,
    "Position Change": position_change,
    "Position Used": position.shift(1),
    "Strategy Return": strategy_returns,
    "Strategy Growth": strategy_growth,
    "Benchmark Growth": benchmark_growth,
    "Biased Growth": biased_growth
})

print(f"Strategy total return: {strategy_total_return:.2%}")
print(f"Benchmark total return: {benchmark_total_return:.2%}")
print(f"Biased total return: {biased_total_return:.2%}")

print(position.value_counts())
print(position_change.value_counts())

print(backtest.loc["2020-06-01":"2020-06-10"])

plt.plot(backtest["Strategy Growth"], label="Momentum Strategy", alpha = 0.7)
plt.plot(backtest["Benchmark Growth"], label="Buy and Hold Benchmark", alpha = 0.7, color="black")
plt.plot(backtest["Biased Growth"], label="Biased Strategy", alpha = 0.7, color="red")
plt.title("Momentum Strategy vs SPY")
plt.ylabel("Growth of $1")
plt.xlabel("Date")
plt.legend()
plt.show()