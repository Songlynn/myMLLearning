import pandas as pd
import numpy as np

# sigmoid函数
def sigmoid(z):
    return (1.0 / (1 + np.exp(-z)))

# 损失函数
def costFunction(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    
    J = (- 1.0 / m) * (np.log(h).T.dot(y) + np.log(1-h).T.dot(1-y))
    
    if np.isnan(J[0]):
        return (np.inf)
    return J[0]

# 损失函数（正则化）
def costFunctionReg(theta, reg, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    
    J = (- 1.0 / m) * (np.log(h).T.dot(y) + np.log(1-h).T.dot(1-y)) + (reg / (2 * m)) * np.sum(np.square(theta[1:]))
    
    if np.isnan(J[0]):
        return (np.inf)
    return J[0]

# 求偏导
def gradient(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1, 1)))
    
    grad = (1.0 / m) * X.T.dot(h-y)
    
    return (grad.flatten())

# 求偏导（正则化）
def gradientReg(theta, reg, X, y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1, 1)))
    
    grad = (1.0 / m) * X.T.dot(h-y) + (reg / m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]
    
    return (grad.flatten())

# 梯度下降
def gradientDescent(theta, X, y, alpha=0.001, num_iters=1500):
    m = y.size
    J_history = np.zeros(num_iters)
    
    for iter in np.arange(num_iters):
        theta = theta - alpha * gradient(theta, X, y)
        J_history[iter] = costFunction(theta, X, y)
    return (theta, J_history)

# 预测
def predict(theta, X, threshold=0.5):
    p = sigmoid(X.dot(theta.T)) >= threshold
    return(p.astype('int'))
