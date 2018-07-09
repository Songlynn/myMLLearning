import numpy as np

def costFunction(X, y, theta=[[0],[0]]):
    m = y.size
    J = 0
    
    h = X.dot(theta)
    J = 1.0 / (2 * m) * (np.sum(np.square(h-y)))

    return J

def gradientDescent(X, y, theta=[[0],[0]], alpha=0.01, num_iter=1500):
    m = y.size
    J_history = np.zeros(num_iter)

    for iter in np.arange(num_iter):
        J_history[iter] = costFunction(X, y, theta)
        h = X.dot(theta)
        theta = theta - alpha * (1.0 / m) * X.T.dot(h-y)

    return (theta, J_history)
