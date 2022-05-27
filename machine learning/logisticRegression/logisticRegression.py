import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('ex2data1.txt', names=['Exam1', 'Exam2', 'Accepted'])



def get_XY(data):
    data.insert(0, 'ones', 1)
    X = data.iloc[:, 0:-1]
    X = X.values
    Y = data.iloc[:, -1]
    Y = Y.values.reshape(len(Y), 1)
    return X,Y

X, Y = get_XY(data)
print(X.shape)
print(Y.shape)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

# define cost function
def costFunction(X, Y,theta):
    A = sigmoid(X@theta)
    first = Y * np.log(A)
    second = (1-Y)*np.log(1-A)
    return -np.sum(first+second) / len(X)

theta = np.zeros((3, 1))
cost_init = costFunction(X, Y, theta)
print(cost_init)


def gradientDescent(X, Y, theta, alpha, iters, isprint=False):
    m = len(X)
    costs = []
    for i in range(iters):
        A = sigmoid(X@theta)
        theta = theta - (alpha/m) * X.T@(A-Y)
        cost = costFunction(X, Y, theta)
        costs.append(cost)

        if i % 1000 == 0:
            if isprint:
                print(cost)
    return costs, theta

alpha = 0.004
iters = 200000
costs, theta_final = gradientDescent(X, Y, theta, alpha, iters)

coef1 = -theta_final[0,0]/theta_final[2,0]
coef2 = -theta_final[1,0]/theta_final[2,0]

x = np.linspace(20, 100, 100)
f = coef1 + coef2 * x
fix, ax = plt.subplots()
ax.scatter(data[data['Accepted']==0]['Exam1'], data[data['Accepted']==0]['Exam2'],
           c='r', marker='x', label='y=0')
ax.scatter(data[data['Accepted']==1]['Exam1'], data[data['Accepted']==1]['Exam2'],
           c='b', marker='o', label='y=1')
ax.legend()
ax.set(xlabel='exam1', ylabel='exam2')
ax.plot(x, f, c='g')
plt.show()
