"""
Name: Giang Duong
ID : 014533857
# Homework 5.15 - SVM
"""
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)


print("q15.a")
# x[i], y[i]
X = np.array([[3, 3],
              [3, 4],
              [2, 3],
              [1, 1],
              [1, 3],
              [2, 2]
              ])
# z[i]
z = np.array([1, 1, 1, -1, -1, -1])
C = 2.5
e = 0.00001
b = 0
lambd = np.zeros(len(X))
lambd_bar = np.zeros(len(X))
E = np.zeros(len(X))
b_arr = np.zeros(len(X))

def findE(XCurr):
    result = 0
    for i in range(len(X)):
        result += (lambd[i] * z[i] * np.dot(X[i], XCurr))
    return result + b

# select indices i, j belongs to [1..n] with i =/= j
for iteration in range(10):
    for i in range(len(X)):
        for j in range(len(X)):
            if i == j: continue

            d = 2 * (np.dot(X[i], X[j])) - np.dot(X[i], X[i]) - np.dot(X[j], X[j])
            if abs(d) > e:
                E[i] = findE(X[i]) - z[i]
                E[j] = findE(X[j]) - z[j]
                lambd_bar[i] = lambd[i]
                lambd_bar[j] = lambd[j]
                lambd[j] = lambd[j] - (z[j] * ((E[i] - E[j]) / d))

                l = 0
                h = 0
                if z[i] == z[j]:
                    l = max(0, lambd[i] + lambd[j] - C)
                    h = min(C, lambd[i] + lambd[j])
                else:
                    l = max(0, lambd[j] - lambd[i])
                    h = min(C, C + lambd[j] - lambd[i])

                if lambd[j] > h:
                    lambd[j] = h
                else:
                    if lambd[j] < l:
                        lambd[j] = l

                lambd[i] = lambd[i] + z[i] * z[j] * (lambd_bar[j] - lambd[j])
                b_arr[i] = b - E[i] - z[i] * (lambd[i] - lambd_bar[i]) * np.dot(X[i], X[i]) - z[j] * (
                        lambd[j] - lambd_bar[j]) * np.dot(X[i], X[j])
                b_arr[j] = b - E[j] - z[i] * (lambd[i] - lambd_bar[i]) * np.dot(X[i], X[j]) - z[j] * (
                        lambd[j] - lambd_bar[j]) * np.dot(X[j], X[j])

                if lambd[i] > 0 and lambd[i] < C:
                    b = b_arr[i]
                elif lambd[j] > 0 and lambd[j] < C:
                    b = b_arr[j]
                else:
                    b = (b_arr[i] + b_arr[j]) / 2

print(lambd)
print(b)
bparta = b
for i in range(len(X)):
    plt.plot([X[i][0]], [X[i][1]], 'ro')
    plt.axis([0,5,0,5])
plt.show()

x = np.arange(-10,10,0.1)
y = - x + b
plt.plot(x,y)
plt.show()
for i in range(len(X)):
    print("F[" + str(i) + "] = " + str(findE(X[i])))
    print("z[" + str(i) + "] = " + str(z[i]))

print("q15.b")

iteration = 0
while iteration < 1000:
    for i in range(len(X)):
        j = rnd.randint(0, 5)
        if i == j: continue
        iteration += 1

        d = 2 * (np.dot(X[i], X[j])) - np.dot(X[i], X[i]) - np.dot(X[j], X[j])
        if abs(d) > e:
            E[i] = findE(X[i]) - z[i]
            E[j] = findE(X[j]) - z[j]
            lambd_bar[i] = lambd[i]
            lambd_bar[j] = lambd[j]
            lambd[j] = lambd[j] - (z[j] * ((E[i] - E[j]) / d))

            l = 0
            h = 0
            if z[i] == z[j]:
                l = max(0, lambd[i] + lambd[j] - C)
                h = min(C, lambd[i] + lambd[j])
            else:
                l = max(0, lambd[j] - lambd[i])
                h = min(C, C + lambd[j] - lambd[i])

            if lambd[j] > h:
                lambd[j] = h
            else:
                if lambd[j] < l:
                    lambd[j] = l

            lambd[i] = lambd[i] + z[i] * z[j] * (lambd_bar[j] - lambd[j])
            b_arr[i] = b - E[i] - z[i] * (lambd[i] - lambd_bar[i]) * np.dot(X[i], X[i]) - z[j] * (
                    lambd[j] - lambd_bar[j]) * np.dot(X[i], X[j])
            b_arr[j] = b - E[j] - z[i] * (lambd[i] - lambd_bar[i]) * np.dot(X[i], X[j]) - z[j] * (
                    lambd[j] - lambd_bar[j]) * np.dot(X[j], X[j])

            if lambd[i] > 0 and lambd[i] < C:
                b = b_arr[i]
            elif lambd[j] > 0 and lambd[j] < C:
                b = b_arr[j]
            else:
                b = (b_arr[i] + b_arr[j]) / 2
    if iteration > 1000:
        break

print(lambd)
print(b)

print("15.c. ")
print("Part a, we have b =", bparta, "so the eqs for hyperplane is x + y =  ", bparta);
print("Part b, we have b =", b, "so the eqs for hyperplane is x + y =  ", b);
for i in range(len(X)):
    plt.plot([X[i][0]], [X[i][1]], 'ro')
    plt.axis([0,5,0,5])
plt.show()

x = np.arange(-10,10,0.1)
y = - x + b
plt.plot(x,y)
plt.show()
