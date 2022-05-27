import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.svm import SVC

data = sio.loadmat('ex6data1.mat')
# print(data.keys())
X,y = data['X'], data['y']
print(X)
def plot_data():
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='jet')
    plt.xlabel('x1')
    plt.ylabel('y1')

plot_data()
# plt.show()svc
svc1 = SVC(C=1, kernel="linear")
svc1.fit(X, y.flatten())
svc1.predict(X)
score = svc1.score(X, y.flatten())
print(score)
def plot_boundary(model):
    x_min, x_max = -0.5, 4.5
    y_min, y_max = 1.3, 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    z = model.predict(np.c_[xx.flatten(), yy.flatten()])
    zz = z.reshape(xx.shape)
    plt.contour(xx, yy, zz)
    plt.show()

plot_boundary(svc1)
plot_data()

svc100 = SVC(C=100, kernel='linear')
svc100.fit(X, y.flatten())
svc100.predict(X)
score = svc100.score(X, y.flatten())
print(score)
plot_boundary(svc100)
plot_data()