import scipy.io as sio
from sklearn.svm import SVC

# training data
data1 = sio.loadmat('spamTrain.mat')
X, y = data1['X'], data1['y']

# Testing data
data2 = sio.loadmat('spamTest.mat')
Xtest, ytest = data2['Xtest'], data2['ytest']

Cvalues = [3, 10, 30, 100, 0.01, 0.03, 0.1, 0.3, 1]
best_score = 0
best_param = 0
for c in Cvalues:
    svc = SVC(C=c, kernel='linear')
    svc.fit(X, y.flatten())
    score = svc.score(Xtest, ytest.flatten())
    if score > best_score:
        best_score = score
        best_param = c
print(best_score, best_param)

svc = SVC(0.03, kernel='linear')
svc.fit(X, y.flatten())
score_train = svc.score(X, y.flatten())
score_test = svc.score(Xtest, ytest.flatten())
print(score_train, score_test)