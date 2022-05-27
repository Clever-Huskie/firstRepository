# 对颜色进行聚类

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage import io

# 1.获取每个样本所属的类别
def find_centroids(X, centros):
    idx = []
    for i in range(len(X)):
        dist = np.linalg.norm((X[i] - centros), axis=1)
        id_i = np.argmin(dist)
        idx.append(id_i)

    return np.array(idx)

# 2.计算聚类中心点
def compute_centros(X, idx, k):
    centros = []
    for i in range(k):
        centros_i = np.mean(X[idx == i], axis=0)  # 按行计算每一列的均值
        centros.append(centros_i)
    return np.array(centros)

# 3.运行kmeans,重复执行1和2
def run_kmeans(X, centros, iters):
    k = len(centros)
    centros_all = []
    centros_all.append(centros)
    centros_i = centros
    for i in range(iters):
        idx = find_centroids(X, centros_i)
        centros_i = compute_centros(X, idx, k)
        centros_all.append(centros_i)

    return idx, np.array(centros_all)

# 观察初试聚类点的位置对聚类效果的影响
def init_centros(X, k):
    index = np.random.choice(len(X), k)
    return X[index]

data = sio.loadmat('bird_small.mat')
A = data['A']

image = io.imread('bird_small.png')
plt.imshow(image)
plt.show()

A = A / 255
A = A.reshape(-1, 3)
k = 16
idx, centros_all = run_kmeans(A, init_centros(A, k=16), iters=20)
centros = centros_all[-1]
im = np.zeros(A.shape)
for i in range(k):
    im[idx == i] = centros[i]
im = im.reshape(128, 128, 3)
plt.imshow(im)
plt.show()
