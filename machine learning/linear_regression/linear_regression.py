import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read data
data = pd.read_csv('ex1data1.txt', names=['population', 'profit'])
data.insert(0, 'ones', 1)
data_head = data.head()
# print(data_head)
# print(data_head)
# print(data.tail())
# print(data.describe())
# print(data.info())
# data.plot.scatter('population', 'profit', label='population')
# plt.show()
X = data.iloc[:, 0: -1]

Y = data.iloc[:, -1]
# print(Y)
# print(Y.head())

# data to matrix
X = X.values
# print(X)
Y = Y.values
Y = Y.reshape(97, 1)

# print(Y)
# print(Y.shape)

def costFunction(X, Y, theta):
    inner = np.power(X @ theta - Y, 2)
    return np.sum(inner) / (2 * len(X))

theta = np.zeros((2, 1))
cost_init = costFunction(X, Y, theta)
# print(cost_init)

def gradientDescent(X, Y, theta, alpha, iters):
    costs = []
    for i in range(iters):
        theta = theta - (X.T @ (X @ theta - Y)) * alpha/len(X)
        cost = costFunction(X, Y, theta)
        costs.append(cost)

        if i %100 == 0:
            print(cost)
    return theta, costs
alpha = 0.02
iters= 2000
theta, costs = gradientDescent(X, Y, theta, alpha, iters)
fig, ax = plt.subplots(1, 2)
ax[0].plot(np.arange(iters), costs, 'r')
ax[0].set(xlabel = 'iters', ylabel = 'cost', title = 'cost vs iters')

# 绘制拟合直线
x = np.linspace(Y.min(), Y.max(), 100)
y = theta[0, 0] + theta[1, 0] * x
ax[1].scatter(X[:, 1], Y, label='training data')
ax[1].plot(x, y, 'r', label='predict')
ax[1].legend()
ax[1].set(xlabel='population', ylabel='profit', title='linearRegression')
plt.show()



