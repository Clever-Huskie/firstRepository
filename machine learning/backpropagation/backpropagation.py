import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.optimize import minimize

data = sio.loadmat('ex4data1.mat')
raw_X = data['X']
raw_y = data['y']

X = np.insert(raw_X, 0, values=1, axis=1)
# print(X)

# 对y进行独热编码处理：one-hot编码
def one_hot_encoder(raw_y):
    result = []
    for i in raw_y:
        y_temp = np.zeros(10)
        y_temp[i-1] = 1

        result.append(y_temp)
    return np.array(result)

y = one_hot_encoder(raw_y)
theta = sio.loadmat('ex4weights.mat')
theta1, theta2 = theta['Theta1'], theta['Theta2']
print(theta1.shape)
print(theta2.shape)


# 序列化权重参数
def serialize(a, b):
    return np.append(a.flatten(), b.flatten())

theta_serialize = serialize(theta1, theta2)
print(theta_serialize.shape)