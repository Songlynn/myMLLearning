# 考试是否通过分类

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import methods as m
import LogisticRegression as LR

data = m.loadData('logistic_regression_data1.txt', ',')

X = np.c_[np.ones((data.shape[0], 1)), data[:, :2]]
y = np.c_[data[:, 2]]

m.plotData(data, 'Exam 1 score', 'Exam 2 score', 'Pass', 'Fail')
plt.show()

initial_theta = np.zeros(X.shape[1])
cost = LR.costFunction(initial_theta, X, y)
grad = LR.gradient(initial_theta, X, y)
print('Cost: ', cost)
print('Grad: ', grad)

# 自己写的梯度下降
theta, cost_J = LR.gradientDescent(initial_theta, X, y)
print('theta: ', theta.ravel())
print('Cost：', cost_J[-1])

m.plotLine(cost_J, 'cose J', 'Iterations')
plt.show()


# 使用scipy的minimize函数
res = minimize(LR.costFunction, initial_theta, args=(X,y), jac=LR.gradient, options={'maxiter':400})
theta = res.x
print('theta: ', theta.ravel())

# 预测：假设考试1得45， 考试2得85
flag = LR.predict(res.x.T, np.array([1, 45, 85]))
p = LR.sigmoid(np.array([1, 45, 85]).dot(theta.T))
print("是否通过：", flag == 1)
print("通过概率：", p)

m.plotBolder(theta, data, 'Exam 1 score', 'Exam 2 score', 'Pass', 'Fail')
plt.show()

