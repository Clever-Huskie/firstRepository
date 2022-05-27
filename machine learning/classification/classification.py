import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.optimize import minimize

data = sio.loadmat('ex3data1.mat')
print(data.keys())
raw_X = data['X']
raw_Y = data['y']
print(raw_X)
def plot_an_image(X):
    pick_one = np.random.randint(5000)
    image = X[pick_one, :]
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.imshow(image.reshape(20, 20).T, cmap='gray_r')
    plt.xticks([])
    plt.yticks([])

def plot_100_image(X):

    sample_index = np.random.choice(len(X), 100)
    images = X[sample_index, :]
    print(images.shape)
    fig, ax = plt.subplots(ncols=10, nrows=10, figsize=(8, 8),
                           sharex=True, sharey=True)

    for r in range(10):
        for c in range(10):
            ax[r, c].imshow(images[10 * r + c].reshape(20, 20).T,
                            cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()

plot_100_image(raw_X)


# loss function
def sigmoid(z):
    return 1/(1 + np.exp(-z))

# define cost function
def costFunction(theta, X, Y, lamda):
    A = sigmoid(X@theta)
    first = Y * np.log(A)
    second = (1-Y)*np.log(1-A)
    # reg = np.sum(np.power(theta[1:], 2))*(lamda/(2*len(X)))
    reg = theta[1:]@theta[1:]*(lamda/(2*len(X)))
    return -np.sum(first+second) / len(X) + reg

# def gradientDescent(X, Y, theta, alpha, iters, lamda, isprint=False):
#     # m = len(X)
#     costs = []
#     for i in range(iters):
#         reg = theta[1:]*(lamda / len(X))
#         reg = np.insert(reg, 0, values=0, axis=0)
#
#         theta = theta - (X.T @ (X @ theta - Y)) * alpha/len(X) - reg
#         cost = costFunction(X, Y, theta, lamda)
#         costs.append(cost)
#
#         if i % 1000 == 0:
#             if isprint:
#                 print(cost)
#     return theta, costs

def gradient_reg(theta, X, Y, lamda):
    reg = theta[1:] * (lamda / len(X))
    reg = np.insert(reg, 0, values=0, axis=0)
    first = (X.T@(sigmoid(X@theta)-Y)) / len(X)
    return first + reg

X = np.insert(raw_X, 0, values=1, axis=1)
print(X.shape)
Y = raw_Y.flatten()

# 优化函数
def one_vs_all(X, Y, lamda, K):
    n = X.shape[1]
    theta_all = np.zeros((K, n))
    for i in range(1, K+1):
        theta_i = np.zeros(n,)
        res = minimize(fun=costFunction, x0=theta_i, args=(X,Y == i,lamda),
                       method='TNC', jac=gradient_reg)
        theta_all[i-1, :] = res.x

    return theta_all
lamda = 1
K = 10
theta_final = one_vs_all(X, Y, lamda, K)

# predict
def predict(X, theta_final):
    h = sigmoid(X@theta_final.T) # (5000,401)(10,401)=>(5000,10)
    h_argmax = np.argmax(h, axis=1)
    return h_argmax + 1

y_pred = predict(X, theta_final)
acc = np.mean(y_pred == Y)
print(acc)
