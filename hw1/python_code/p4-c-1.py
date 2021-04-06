import pandas as pd
import  numpy as np
X = pd.read_csv("Sample data of X.csv", header=0, index_col=0)
y = pd.read_csv("Sample data of y.csv", header=None)
theta_star = pd.read_csv("data of theta_star.csv", header=None)


X = np.array(X)
y = np.array(y)
theta_star = np.array(theta_star)

theta_LS = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
error = np.linalg.norm(theta_star - theta_LS, ord = 2)
print(error)