import numpy as np
prices = np.array([100, 103, 101, 106, 108])
print("Prices: ", prices)
print("First price: ", prices[0])
print("Last Price: ", prices[-1])
print("Middle three prices: ", prices[1:4])
print("10% Increase on prices: ", prices * 1.1)
print("Mean: ", np.mean(prices))
print("Standard Deviation: ", np.std(prices))