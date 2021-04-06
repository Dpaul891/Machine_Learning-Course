import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def hu(z):
    if abs(z) >= mu:
        return abs(z)
    else:
        return z*z/(2*mu) + mu/2
def f(theta):
    return sum(map(hu, np.dot(X, theta) - y))[0]

def df(theta):
    t = np.dot(X, theta) - y
    t1 = np.dot(X, theta) - y
    t[abs(t)>=mu] = abs(t)[abs(t)>=mu]
    t[abs(t)<=mu] = mu
    return np.dot(X.T, t1/t)

def GM(theta):
    thetak = theta
    norm_list = []
    norm_list.append(np.linalg.norm(thetak-theta_star))
    for k in np.arange(T):
        thetak = thetak - alpha * df(thetak)
        norm_list.append(np.linalg.norm(thetak-theta_star))
    norm_list = np.array(norm_list)  
    plot_convergence(norm_list)
    return thetak

def plot_convergence(norm_list):
    number = norm_list.size
    x = np.arange(number)
    #y = np.log(norm_list)
    y = norm_list
    plt.plot(x, y, linewidth=1)
    plt.xlabel('number of iterations')
    plt.ylabel(r'$||\theta_k-\theta^{\star}||_2$')
    plt.tight_layout()
    plt.savefig('./plt.pdf', dpi=1000)
      
X = pd.read_csv("Sample data of X.csv", header=0, index_col=0)
y = pd.read_csv("Sample data of y.csv", header=None)
theta_star = pd.read_csv("data of theta_star.csv", header=None)

X = np.array(X)
y = np.array(y)
theta_star = np.array(theta_star)

n = X.shape[0]
d = X.shape[1]
mu = 1e-5
alpha = 1e-3
T = 1000
theta0 = np.zeros(d).reshape(d,1)
thetaT = GM(theta0)
 

