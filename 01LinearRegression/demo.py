import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import LinearRegression as lr

# 1.数据处理
data = np.loadtxt('linear_regression_data1.txt', delimiter=',')
print('Data:')
print(data[0:5])

X = np.c_[np.ones(data.shape[0]), data[:, 0]]
y = np.c_[data[:, 1]]

# 2.绘制散点图
plt.scatter(X[:, 1], y,s=30, c='r', marker='x', linewidths=1)
plt.xlim(4,24)
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.show()

# 3.绘制损失函数迭代变化
theta, cost_J = lr.gradientDescent(X, y)
print('theta: ', theta.ravel())

plt.plot(cost_J)
plt.ylabel('cost J')
plt.xlabel('Iterations')
plt.show()

# 4.与sklearn的线性回归模型对比
xx = np.arange(5,23)
yy = theta[0]+theta[1]*xx

# 画出我们自己写的线性回归梯度下降收敛的情况
plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
plt.plot(xx,yy, label='Linear regression (Gradient descent)')
# 和Scikit-learn中的线性回归对比一下 
regr = LinearRegression()
regr.fit(X[:,1].reshape(-1,1), y.ravel())
plt.plot(xx, regr.intercept_+regr.coef_*xx, label='Linear regression (Scikit-learn GLM)')

plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.legend(loc=4);
plt.show()
