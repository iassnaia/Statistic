'''Даны значения величины заработной платы заемщиков банка (zp) и значения их поведенческого кредитного скоринга (ks): 
zp = [35, 45, 190, 200, 40, 70, 54, 150, 120, 110], 
ks = [401, 574, 874, 919, 459, 739, 653, 902, 746, 832]. 
Используя математические операции, посчитать коэффициенты линейной регрессии, приняв за X заработную плату 
(то есть, zp - признак), а за y - значения скорингового балла (то есть, ks - целевая переменная). 
Произвести расчет как с использованием intercept, так и без.'''


import numpy as np
import scipy.stats as stats 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])
n = 10

b1 = (n*np.sum(zp*ks) - np.sum(zp)*np.sum(ks)) / \
    (n*np.sum(zp**2) - np.sum(zp)**2)

#по второму способу
b1_1 = (np.mean(zp*ks) - np.mean(zp)*np.mean(ks)) / \
    (np.mean(zp**2) - np.mean(zp)**2)

b0 = np.mean(ks) - b1*np.mean(zp)


ks_pred = b0 + b1 * zp
print(b1, b1_1, b0, ks_pred)

def mse_(B1, y = ks, x = zp, n = 10):
    return np.sum((B1*x - y)**2)/n

alpha = 1e-6
B1 = 0.1

for i in range (3000):
    B1 -= alpha * (2/n) * np.sum ((B1 * zp - ks) * zp)
    if i % 500 == 0:
        print ('Iteration = {i}, B1 = {B1}, mse = {mse}'.format(i = i, B1 = B1, mse = mse_(B1)))

def mse_1(B1, B0, y = ks, x = zp, n = 10):
    return np.sum((B0 + B1*x - y)**2)/n

alpha = 1e-6
B0 = 0.1
B1 = 0.1

for i in range (3000):
    B1 -= alpha * (2/n) * np.sum ((B0 + B1 * zp - ks) * zp)
    B0 -= alpha * (2/n) * np.sum ((B0 + B1 * zp - ks) * zp)
    if i % 500 == 0:
        print ('Iteration = {i}, B1 = {B1}, B0 = {B0}, mse = {mse}'.format(i = i, B1 = B1, B0 = B0, mse = mse_1(B1, B0)))
