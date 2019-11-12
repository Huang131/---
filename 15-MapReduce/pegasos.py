'''
@version: 0.0.1
@Author: Huang
@dev: python3 vscode
@Date: 2019-11-12 23:23:44
@LastEditTime: 2019-11-12 23:36:23
@FilePath: \\机器学习实战\\15-MapReduce\\pegasos.py
@Descripttion: 
将w初始化为0
对每次批处理
    随机选择k个样本点(向量)
    对每个向量
        如果该向量被错分:
            更新权重向量w
    累加对w的更新
'''

import numpy as np
import matplotlib.pyplot as plt


# 加载数据
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        # dataMat.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2])])
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def seqPegasos(dataSet, labels, lam, T):
    m, n = np.shape(dataSet)
    w = np.zeros(n)
    for t in range(1, T + 1):
        i = np.random.randint(m)
        eta = 1.0 / (lam * t)
        p = predict(w, dataSet[i, :])
        if labels[i] * p < 1:
            w = (1.0 - 1 / t) * w + eta * labels[i] * dataSet[i, :]
        else:
            w = (1.0 - 1 / t) * w
        print(w)
    return w


def predict(w, x):
    return w * x.T


def batchPegasos(dataSet, labels, lam, T, k):
    m, n = np.shape(dataSet)
    w = np.zeros(n)
    dataIndex = range(m)
    for t in range(1, T + 1):
        wDelta = np.mat(np.zeros(n))  # reset wDelta
        eta = 1.0 / (lam * t)
        np.random.shuffle(dataIndex)
        for j in range(k):  # go over training set
            i = dataIndex[j]
            p = predict(w, dataSet[i, :])  # mapper code
            if labels[i] * p < 1:  # mapper code
                wDelta += labels[i] * dataSet[i, :].A  # accumulate changes
        w = (1.0 - 1 / t) * w + (eta / k) * wDelta  # apply changes at each T
    return w


datArr, labelList = loadDataSet(r'15-MapReduce\testSet.txt')
datMat = np.mat(datArr)
# finalWs = seqPegasos(datMat, labelList, 2, 5000)
finalWs = batchPegasos(datMat, labelList, 2, 50, 100)
print(finalWs)

fig = plt.figure()
ax = fig.add_subplot(111)
x1 = []
y1 = []
xm1 = []
ym1 = []
for i in range(len(labelList)):
    if labelList[i] == 1.0:
        x1.append(datMat[i, 0])
        y1.append(datMat[i, 1])
    else:
        xm1.append(datMat[i, 0])
        ym1.append(datMat[i, 1])
ax.scatter(x1, y1, marker='s', s=90)
ax.scatter(xm1, ym1, marker='o', s=50, c='red')
x = np.arange(-6.0, 8.0, 0.1)
y = (-finalWs[0, 0] * x - 0) / finalWs[0, 1]
# y2 = (0.43799*x)/0.12316
y2 = (0.498442 * x) / 0.092387  # 2 iterations
ax.plot(x, y)
ax.plot(x, y2, 'g-.')
ax.axis([-6, 8, -4, 5])
ax.legend(('50 Iterations', '2 Iterations'))
plt.show()
