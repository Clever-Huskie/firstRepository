import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('ex1data2.txt', names=['size', 'bedrooms', 'prices'])
# print(data.head())
def normalize_feature(data):
    return (data - data.mean()) / data.std()
data = normalize_feature(data)
print(data.head())
# 添加全为1的一列
data.insert(0, 'ones', 1)
X = data.iloc[:, 0:-1]
Y = data.iloc[:, -1]
X = X.values
Y = Y.values
Y = Y.reshape(47, 1)

def costFunction(X, Y, theta):
    inner = np.power(X @ theta - Y, 2)
    return np.sum(inner) / (2 * len(X))

theta = np.zeros((3, 1))
cost_init = costFunction(X, Y, theta)
# print(cost_init)

def gradientDescent(X, Y, theta, alpha, iters, isprint=False):
    costs = []
    for i in range(iters):
        theta = theta - (X.T @ (X @ theta - Y)) * alpha/len(X)
        cost = costFunction(X, Y, theta)
        costs.append(cost)

        if i %100 == 0:
            if isprint:
                print(cost)
    return theta, costs

theta = np.zeros((3, 1))
cost_init = costFunction(X, Y, theta)
print(cost_init)

# 不同alpha下的效果
iters = 2000
candinate_alpha = [0.0003, 0.003, 0.03, 0.0001, 0.001, 0.01]

# 绘制图像
fig, ax = plt.subplots()
for alpha in candinate_alpha:
    _, costs = gradientDescent(X, Y, theta, alpha, iters)
    ax.plot(np.arange(iters), costs)


ax.set(xlabel='iters', ylabel='cost', title='cost vs iters')
plt.show()