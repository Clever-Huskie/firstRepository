import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('ex1data1.txt', names=['population', 'profit'])
data.insert(0, 'ones', 1)
X = data.iloc[:, 0: -1]

Y = data.iloc[:, -1]
X = X.values
Y = Y.values
Y = Y.reshape(97, 1)
def normalEquation(X, Y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ Y
    return theta

theta = normalEquation(X, Y)
print(theta)