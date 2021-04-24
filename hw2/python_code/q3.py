#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 22:12:10 2021

@author: dpaul891
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#a1------------
data = pd.read_csv("Q3_training_data.csv", header=0, index_col=0)
x_train = np.array(data.iloc[0,:]).reshape(-1, 1)
y_train = np.array(data.iloc[1,:]).reshape(-1, 1)

#a2------------
x_sim = np.arange(-1.5, 1.5, 0.01)
y_sim = x_sim ** 2

plt.figure(1)
plt.plot(x_sim, y_sim, color="brown", label="target function")
plt.scatter(x_train, y_train, marker='o', c='', edgecolors='b', label="data")
plt.xlim(-1.5, 1.5)
plt.ylim(-0.5, 2.5)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend(loc="best", edgecolor="black")
plt.savefig("hw_tex/tupian/original.pdf")

order = 8
n_train = 10
x_train_new = np.ones(n_train).reshape(-1,1)
new_col = np.ones(n_train).reshape(-1,1)
for i in range(1,order+1):
    new_col = new_col * x_train
    x_train_new = np.hstack((x_train_new, new_col))
y_train_new = y_train
theta_hat = np.dot(np.dot(np.linalg.inv(np.dot(x_train_new.T, x_train_new)), x_train_new.T), y_train_new)

x_sim = np.arange(-1.5, 1.5, 0.01)
y_sim = np.zeros(len(x_sim))
for i in range(9):
    y_sim = y_sim + theta_hat[i] * (x_sim**i)

plt.plot(x_sim, y_sim, label="fitted curve")
plt.legend(loc="best", edgecolor="black")
plt.savefig("hw_tex/tupian/fitted.pdf")

#a3------------
data_test = pd.read_csv("Q3_test_data.csv", header=0, index_col=0)
x_test = np.array(data_test.iloc[0,:]).reshape(-1, 1)
y_test = np.array(data_test.iloc[1,:]).reshape(-1, 1)

n_test = 10
x_test_new = np.ones(n_test).reshape(-1,1)
new_col = np.ones(n_test).reshape(-1,1)
for i in range(1,order+1):
    new_col = new_col * x_test
    x_test_new = np.hstack((x_test_new, new_col))
y_test_new = y_test
test_error = np.linalg.norm(np.dot(x_test_new, theta_hat) - y_test_new, ord=2)
print("test error:", test_error)

#b1------------
lambda_candidate = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.8, 1, 2, 5, 10, 15, 20, 50, 100]
k = 5
s_size = n_train / k
error_list = []
for lambda_i in lambda_candidate:
    error = 0
    for val_i in range(k):
        x_val = x_train_new[[val_i*2, val_i*2+1], :]
        y_val = y_train_new[[val_i*2, val_i*2+1], :]
        x_train_val = np.vstack((x_train_new[0:val_i*2, :], x_train_new[val_i*2+2:, :]))
        y_train_val = np.vstack((y_train_new[0:val_i*2, :], y_train_new[val_i*2+2:, :]))
        I = np.identity(x_train_val.shape[1])
        theta_hat = np.dot(np.dot(np.linalg.inv(np.dot(x_train_val.T, x_train_val)+lambda_i*I), x_train_val.T), y_train_val)
        error  = error + np.linalg.norm(np.dot(x_val, theta_hat) - y_val, ord=2)

    error = error / k
    error_list.append(error)

plt.figure(2)
plt.semilogx(lambda_candidate, error_list, linewidth=2, color="purple", label="validation error")
plt.xlabel("$\lambda$")
plt.ylabel("validation error")
plt.legend(loc="best", edgecolor="black")
plt.savefig("hw_tex/tupian/validation.pdf")

#b2------------
lambda_candidate = [0.01, 0.1, 0.8, 5]
figure_num = 1
plt.figure(3, figsize=(20, 20))

test_error_list = []
for lambda_i in lambda_candidate:
    plt.subplot(2, 2, figure_num)
    figure_num = figure_num + 1
    
    x_sim = np.arange(-1.5, 1.5, 0.01)
    y_sim = x_sim ** 2
    plt.plot(x_sim, y_sim, color="brown", label="target function")
    plt.scatter(x_train, y_train, marker='o', c='', edgecolors='b', label="data")

    
    I = np.identity(x_train_new.shape[1])
    theta_hat = np.dot(np.dot(np.linalg.inv(np.dot(x_train_new.T, x_train_new)+lambda_i*I), x_train_new.T), y_train_new)
    x_sim = np.arange(-1.5, 1.5, 0.01)
    y_sim = np.zeros(len(x_sim))
    for i in range(9):
        y_sim = y_sim + theta_hat[i] * (x_sim**i)
    plt.plot(x_sim, y_sim, label="fitted curve")
    
    plt.xlim(-1.5, 1.5)
    plt.ylim(-0.5, 2.5)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("$\lambda$={}".format(lambda_i))
    plt.legend(loc="lower left", edgecolor="black")
    plt.tight_layout()
    
    test_error = np.linalg.norm(np.dot(x_test_new, theta_hat) - y_test_new, ord=2)
    test_error_list.append(test_error)
    
plt.savefig("hw_tex/tupian/lambda.pdf")

#b3------------
print()
print("-"*20)
for i, test_error in enumerate(test_error_list):
    print("lambda={}, test error={}".format(lambda_candidate[i], test_error))
    

    
    
        
        
        
        



















