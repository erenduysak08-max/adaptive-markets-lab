import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
returns = pd.Series([
    0.001,
    0.002,
    -0.001,
    0.001,
    0.000,
    0.002,
    0.001,
    0.002,
    0.050,
    0.060,
    0.040
])
fast = returns.ewm(halflife=2).mean()
slow = returns.ewm(halflife=10).mean()

plt.plot(returns, label="Returns")
plt.plot(fast, label="Fast EWM")
plt.plot(slow, label="Slow EWM")
plt.plot(y_values)
plt.legend()
plt.show()