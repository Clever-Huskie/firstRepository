import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.svm import SVC

mat = sio.loadmat('ex6data3.mat')

X, y = mat['X'], mat['y']
Xval, yval = mat['Xval'], mat['yval']
def plot_data():
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='jet')
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.show()

plot_data()
Cvalues = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gammas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

best_score = 0
best_params = (0, 0)
for c in Cvalues:
    for gamma in gammas:
        svc = SVC(C=c, kernel='rbf', gamma=gamma)
        svc.fit(X, y.flatten())
        score = svc.score(Xval, yval.flatten())
        if score > best_score:
            best_score = score
            best_params = (c, gamma)

print(best_score, best_params)
svc2 = SVC(C=0.3, kernel='rbf', gamma=100)
svc2.fit(X, y.flatten())

def plot_boundary(model):
    x_min, x_max = -0.6, 0.4
    y_min, y_max = -0.7, 0.6
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    z = model.predict(np.c_[xx.flatten(), yy.flatten()])
    zz = z.reshape(xx.shape)
    plt.contour(xx, yy, zz)

plot_boundary(svc2)
plot_data()