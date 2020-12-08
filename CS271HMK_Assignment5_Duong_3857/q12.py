"""
Name: Giang Duong
ID : 014533857
# Homework 5.12- SVM
"""
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
print("q12./")
# read file into array
p = pd.read_csv("malwareBenignScores.csv")

# Separate file into 2 parts
X = p[['HMM', 'SSD', 'OGS']]
Y = p[['SCORE']]
# get data into array
X_train = X[:41]        # 20 begins and 20 malware sample
Y_train = Y[:41]        # 20 begins and 20 malware sample score
X_test = X[41:]         # test samples
Y_test = Y[41:]         # test samples score
# create an instance of SVM and fit out data.
svc = svm.SVC(kernel='linear')
svc.fit(X_train, Y_train.values.ravel())
# perform a classification on samples in X_test
Y_test_Pred = svc.predict(X_test)
# get accuracy score
acc = accuracy_score(Y_test_Pred, Y_test)
print("With all 3 features")
print(acc)
respective = svc.coef_
print("Respective Weights of HMM, SSD, OGS")
print(respective)
print("The most important weight is OGS score when the least important weight is HMM score ")
# get data into array
X = p[['SSD', 'OGS']]
X_train = X[:41]
Y_train = Y[:41]
X_test = X[41:]
Y_test = Y[41:]
# create an instance of SVM and fit out data.
svc = svm.SVC(kernel='linear')
svc.fit(X_train, Y_train.values.ravel())
# perform a classification on samples in X_test
Y_test_Pred = svc.predict(X_test)
# get accuracy score
acc = accuracy_score(Y_test_Pred, Y_test)
print("Accuracy after removing the HMM feature -> ")
print(acc)
respective = svc.coef_
print("Respective Weights of SSD, OGS")
print(respective)
# get data into array
X = p[['SSD']]
X_train = X[:41]
Y_train = Y[:41]
X_test = X[41:]
Y_test = Y[41:]
# create an instance of SVM and fit out data.
svc = svm.SVC(kernel='linear')
svc.fit(X_train, Y_train.values.ravel())
# perform a classification on samples in X_test
Y_test_Pred = svc.predict(X_test)
# get accuracy score

acc = accuracy_score(Y_test_Pred, Y_test)
print("Accuracy after removing the HMM and OGS feature -> ")
print(acc)
respective = svc.coef_
print("Weight of SSD")
print(respective)
