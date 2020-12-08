"""
Name: Giang Duong
ID : 014533857
# Homework 5.10 - SVM
"""
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score

# read the file
file = pd.read_csv("malwareBenignScores.csv")
X = file[['HMM', 'SSD', 'OGS']]
Y = file[['SCORE']]

X_train = X[:41]
Y_train = Y[:41]
X_test = X[41:]
Y_test = Y[41:]

print("10.a.")
print("C = 1.0, p= 2")
# svm regularization parameter
# create an instance of SVM and fit out data.
clf_svc = svm.SVC(kernel='poly', C=1, degree=2, gamma='auto')

clf_svc.fit(X_train.values, Y_train.values.ravel(order='K'))
# perform a classification on samples in X_test
Y_test_Pred = clf_svc.predict(X_test)
print(Y_test_Pred)
acc = accuracy_score(Y_test, Y_test_Pred)
print(acc)
# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(Y_test,Y_test_Pred))
# print(classification_report(Y_test,Y_test_Pred))

print("\n10.b. ")
print("C = 3.0, p= 2")
# create an instance of SVM and fit out data.
clf_svc = svm.SVC(kernel='poly', C=3.0, degree=2, gamma='auto')
clf_svc.fit(X_train, Y_train.values.ravel())
# perform a classification on samples in X_test
Y_test_Pred = clf_svc.predict(X_test)
print(Y_test_Pred)
acc = accuracy_score(Y_test_Pred, Y_test)
print(acc)

print("\n10.c. ")
print("C = 1.0, p= 4")
# create an instance of SVM and fit out data.
clf_svc = svm.SVC(kernel='poly', C=1.0, degree=4, gamma='auto')
clf_svc.fit(X_train, Y_train.values.ravel())
# perform a classification on samples in X_test
Y_test_Pred = clf_svc.predict(X_test)
print(Y_test_Pred)
acc = accuracy_score(Y_test_Pred, Y_test)
print(acc)

print("\n10.d. ")
print("C = 3.0, p= 4")
# create an instance of SVM and fit out data.
clf_svc = svm.SVC(kernel='poly', C=3.0, degree=4, gamma='auto')
clf_svc.fit(X_train, Y_train.values.ravel())
# perform a classification on samples in X_test
Y_test_Pred = clf_svc.predict(X_test)
print(Y_test_Pred)
acc = accuracy_score(Y_test_Pred, Y_test)
print(acc)
