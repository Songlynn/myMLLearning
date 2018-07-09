import matplotlib.pyplot as plt
import numpy as np

import LogisticRegression as LR

def loadData(file, delimeter):
    data = np.loadtxt(file,delimiter=delimeter)
    print('维度：', data.shape)
    print('数据（前5行）：')
    print(data[:5, :])
    return (data)

## 绘制散点图
def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    # 获取正负样本的下标
    neg = data[:, 2] == 0
    pos = data[:, 2] == 1
    
    if axes == None:
        axes = plt.gca()
    # 绘制散点图
    axes.scatter(data[pos][:, 0], data[pos][:, 1], marker='+', c='k', s=60, linewidth=2,label=label_pos)
    axes.scatter(data[neg][:, 0], data[neg][:, 1], marker='x', c='y', s=60, linewidth=2,label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon=True, fancybox = True);
## 绘制曲线
def plotLine(data, label_x, label_y):
    plt.plot(data)
    plt.ylabel(label_x)
    plt.xlabel(label_y)

def plotBolder(theta, data, label_x, label_y, label_pos, label_neg, axes=None):
    plt.scatter(45, 85, s=60, c='r', marker='v', label='(45, 85)')
    plotData(data, label_x, label_y, label_pos, label_neg, axes)
    X = np.c_[np.ones((data.shape[0], 1)), data[:, :2]]
    x1_min, x1_max = X[:,1].min(), X[:,1].max(),
    x2_min, x2_max = X[:,2].min(), X[:,2].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = LR.sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b');
