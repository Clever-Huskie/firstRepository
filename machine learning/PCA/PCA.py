# 使用PCA进行降维

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

mat = sio.loadmat('ex7data1.mat')
X = mat['X']
plt.scatter(X[:, 0], X[:, 1])
plt.show()

# 1.对X去均值化
X_demean = X - np.mean(X, axis=0)
plt.scatter(X_demean[:, 0], X_demean[:, 1])


# 2.计算协方差矩阵
C = X_demean.T @ X_demean / len(X)

# 3.计算特征值，特征向量
U, S, V = np.linalg.svd(C)

# 4.实现降维
U1 = U[:, 0]
X_reduction = X_demean@U1
plt.plot([0, U1[0]], [0, U1[1]], c='r')
plt.plot([0, U[:, 1][0]], [0, U[:, 1][1]], c='k')
plt.show()

# 5.还原数据
X_restore = X_reduction.reshape(50, 1)@U1.reshape(1, 2)+np.mean(X, axis=0)
plt.scatter(X[:, 0], X[:, 1])
plt.scatter(X_restore[:, 0], X_restore[:, 1])
plt.show()
