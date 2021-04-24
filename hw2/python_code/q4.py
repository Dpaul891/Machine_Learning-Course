#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 14:59:42 2021

@author: dpaul891
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#training data
training_data = pd.read_csv("Q4_MNIST_train.csv", header=0, index_col=0)
x = np.array(training_data.iloc[:, 0])
y = np.array(training_data.iloc[:, 1])
K = 10 #number of class 

for i in range(len(x)):
    x_i = x[i][1:-1].split(',')
    for j in range(len(x_i)):
        x_i[j] = float(x_i[j])
    x_i = np.array(x_i).reshape(1,-1)
    if i==0:
        X_train = x_i
    else:
        X_train = np.vstack((X_train, x_i))

for i in range(len(y)):
    l = int(y[i][1:-1])
    y_i = np.zeros(K)
    y_i[l-1] = 1
    if i == 0:
        y_train = y_i
    else:
        y_train = np.vstack((y_train, y_i))

n = X_train.shape[0]
dim = X_train.shape[1]

#testing data
test_data = pd.read_csv("Q4_MNIST_test.csv", header=0, index_col=0)
x = np.array(test_data.iloc[:, 0])
y = np.array(test_data.iloc[:, 1])

for i in range(len(x)):
    x_i = x[i][1:-1].split(',')
    for j in range(len(x_i)):
        x_i[j] = float(x_i[j])
    x_i = np.array(x_i).reshape(1,-1)
    if i==0:
        X_test = x_i
    else:
        X_test = np.vstack((X_test, x_i))

for i in range(len(y)):
    l = int(y[i][1:-1])
    y_i = np.zeros(K)
    y_i[l-1] = 1
    if i == 0:
        y_test = y_i
    else:
        y_test = np.vstack((y_test, y_i))

#calculate gradient
def cal_grad(theta):
    Z = np.dot(X_train, theta)
    exp_Z = np.exp(Z)
    sum_Z = np.sum(exp_Z, axis=1).reshape(-1, 1)
    vector_Z = exp_Z / sum_Z
    
    grad = np.dot(X_train.T, vector_Z-y_train)
    grad = grad / n
    return grad

def cal_accuracy(theta, X, y):
    total_num = X.shape[0]
    y_predict = np.argmax(np.dot(X, theta), axis=1)
    y = np.argmax(y, axis=1)
    correct_num = np.sum(y_predict == y)
    return correct_num / total_num

def plot_convergence(train_accuracy, test_accuracy):
    iteration_num = len(train_accuracy)
    iterate = range(iteration_num)
    
    plt.figure(1)
    plt.plot(iterate, train_accuracy, color="blue", label="train accuracy")
    plt.plot(iterate, test_accuracy, color="brown", label="test accuracy")
    plt.xlabel("iteration number")
    plt.ylabel("accuracy")
    plt.legend(loc="best", edgecolor="black")
    plt.tight_layout()
    plt.savefig("hw_tex/tupian/accuracy.pdf")
    
def AGD(initial):
    
    mu = 0.01 # step size
    theta_minus = initial
    theta_k = initial
    
    num_iteration = 0
    
    train_accuracy = []
    test_accuracy = []
    
    train_accuracy.append(cal_accuracy(theta_k, X_train, y_train))
    test_accuracy.append(cal_accuracy(theta_k, X_test, y_test))
    while num_iteration <= 500:
        
        beta_k = (num_iteration-1)/(num_iteration+2)
        w = theta_k + beta_k * (theta_k - theta_minus)
        
        theta_minus = theta_k
        theta_k = w - mu * cal_grad(w)
        
        train_accuracy.append(cal_accuracy(theta_k, X_train, y_train))
        test_accuracy.append(cal_accuracy(theta_k, X_test, y_test))
        
        num_iteration  = num_iteration + 1
        
    plot_convergence(train_accuracy, test_accuracy)
    print("train accuracy at the last step:", train_accuracy[-1])
    print("test accuracy at the last step:", test_accuracy[-1])
theta_initial = np.zeros((dim, K))
AGD(theta_initial)

















