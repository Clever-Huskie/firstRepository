import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
'''
逻辑回归-线性不可分案例
案例：设想你是工厂的生产主管，你要决定是否芯片要被接受或抛弃
数据集：ex2data2.txt，芯片在两次测试中的测试结果 
'''

data = pd.read_csv('ex2data2.txt', names=['Test1', 'Test2', 'Accepted'])

fix, ax = plt.subplots()
ax.scatter(data[data['Accepted']==0]['Test1'], data[data['Accepted']==0]['Test2'],
           c='r', marker='x', label='y=0')
ax.scatter(data[data['Accepted']==1]['Test1'], data[data['Accepted']==1]['Test2'],
           c='b', marker='o', label='y=1')
ax.legend()
ax.set(xlabel='Test1', ylabel='Test2')
# plt.show()

# 特征映射
def feature_mapping(x1, x2, power):
    data = {}
    for i in np.arange(power+1):
        for j in np.arange(i + 1):
            data['F{}{}'.format(i-j, j)] = np.power(x1, i-j)*np.power(x2, j)

    return pd.DataFrame(data)

x1 = data['Test1']
x2 = data['Test2']

data2 = feature_mapping(x1, x2, 6)
# print(data2.head())

X = data2.values
# print(X)
# print(X.shape)

Y = data.iloc[:, -1].values
Y = Y.reshape(len(Y), 1)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

# define cost function
def costFunction(X, Y,theta, lamda):
    A = sigmoid(X@theta)
    first = Y * np.log(A)
    second = (1-Y)*np.log(1-A)
    reg = np.sum(np.power(theta[1:], 2))*(lamda/(2*len(X)))
    return -np.sum(first+second) / len(X) + reg

theta = np.zeros((28, 1))
lamda = 1
cost_init = costFunction(X, Y, theta, lamda)
print(cost_init)

def gradientDescent(X, Y, theta, alpha, iters, lamda, isprint=False):
    m = len(X)
    costs = []
    for i in range(iters):
        reg = theta[1:]*(lamda / len(X))
        reg = np.insert(reg, 0, values=0, axis=0)

        theta = theta - (X.T @ (X @ theta - Y)) * alpha/len(X) - reg
        cost = costFunction(X, Y, theta, lamda)
        costs.append(cost)

        if i % 1000 == 0:
            if isprint:
                print(cost)
    return theta, costs

alpha = 0.001
iters = 200000
lamda = 0.001
theta_final, costs = gradientDescent(X, Y, theta, alpha, iters, lamda)

def predict(X, theta):

    prob = sigmoid(X@theta)
    return [1 if x>=0.5 else 0 for x in prob]

y = np.array(predict(X, theta_final))
y_pre = y.reshape(len(y), 1)
acc = np.mean(y_pre == y)
print(acc)

x= np.linspace(-1)

plt.contour(xx, yy, zz, 0)

