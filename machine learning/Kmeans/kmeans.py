import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage import io

data1 = sio.loadmat('ex7data2.mat')
# print(data1)

X = data1['X']
plt.scatter(X[:, 0], X[:, 1])
plt.show()

# 1.获取每个样本所属的类别
def find_centroids(X, centros):
    idx = []
    for i in range(len(X)):
        dist = np.linalg.norm((X[i] - centros), axis=1)
        id_i = np.argmin(dist)
        idx.append(id_i)

    return np.array(idx)

centros = np.array([[3, 3], [6, 2], [8, 5]])
idx = find_centroids(X, centros)

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

# 4.绘制数据集和聚类中心的移动轨迹
def plot_data(X, centros_all, idx):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=idx, cmap='rainbow')
    plt.plot(centros_all[:, :, 0], centros_all[:, :, 1], 'kx--')
    plt.show()

idx, centros_all = run_kmeans(X, centros, iters=10)
plot_data(X, centros_all, idx)

# 观察初试聚类点的位置对聚类效果的影响
def init_centros(X, k):
    index = np.random.choice(len(X), k)
    return X[index]

for i in range(4):
    idx, centros_all = run_kmeans(X, init_centros(X, k=3), iters=10)
    plot_data(X, centros_all, idx)




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
    im[id == i] = centros[i]
im = im.reshape(128, 128, 3)
plt.imshow(im)
plt.show()
