"""
Name: Giang Duong
ID : 014533857
# Homework 5.11 - SVM
"""
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
print("q11./")
# read the file
file = pd.read_csv("malwareBenignScores.csv")
X = file[['HMM', 'SSD', 'OGS']]
Y = file[['SCORE']]

X_train = X[:41]
Y_train = Y[:41]
X_test = X[41:]
Y_test = Y[41:]
C = [1,2, 3,4]
gamma = [2, 3, 4, 5]

for i in C:
    for j in gamma:
        # svm regularization parameter
        # create an instance of SVM and fit out data.
        clf_svc = svm.SVC(kernel='rbf', C=i, gamma=j)
        clf_svc.fit(X_train.values, Y_train.values.ravel(order='K'))
        # perform a classification on samples in X_test
        Y_test_Pred = clf_svc.predict(X_test)
        print(Y_test_Pred)
        acc = accuracy_score(Y_test, Y_test_Pred)
        print(acc)