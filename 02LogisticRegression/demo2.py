# 高维分类——正则化

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures

import methods as m
import LogisticRegression as LR

data = m.loadData('logistic_regression_data2.txt', ',')

X = data[:, :2]
y = np.c_[data[:, 2]]

m.plotData(data, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0')
plt.show()

#  添加多项式
poly = PolynomialFeatures(6)
XX = poly.fit_transform(X)
print("当前维度：", XX.shape)

initial_theta = np.zeros(XX.shape[1])
cost = LR.costFunctionReg(initial_theta, 1, XX, y)
print('Cost: ', cost)


fig, axes = plt.subplots(1,3, sharey = True, figsize=(17,5))

# 决策边界，咱们分别来看看正则化系数lambda太大太小分别会出现什么情况
# Lambda = 0 : 就是没有正则化，这样的话，就过拟合咯
# Lambda = 1 : 这才是正确的打开方式
# Lambda = 100 : 卧槽，正则化项太激进，导致基本就没拟合出决策边界

for i, C in enumerate([0.0, 1.0, 100.0]):
    # 最优化 costFunctionReg
    res = minimize(LR.costFunctionReg, initial_theta, args=(C, XX, y), jac=LR.gradientReg, options={'maxiter':3000})
    
    # 准确率
    accuracy = 100.0*sum(LR.predict(res.x, XX) == y.ravel())/y.size    

    # 对X,y的散列绘图
    m.plotData(data, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0', axes.flatten()[i])
    
    # 画出决策边界
    x1_min, x1_max = X[:,0].min(), X[:,0].max(),
    x2_min, x2_max = X[:,1].min(), X[:,1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = LR.sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(res.x))
    h = h.reshape(xx1.shape)
    axes.flatten()[i].contour(xx1, xx2, h, [0.5], linewidths=1, colors='g');       
    axes.flatten()[i].set_title('Train accuracy {}% with Lambda = {}'.format(np.round(accuracy, decimals=2), C))
plt.show()
