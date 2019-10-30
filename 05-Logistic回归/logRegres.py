'''
@version: 0.0.1
@Author: Huang
@dev: python3 vscode
@Date: 2019-10-29 23:39:44
@LastEditTime: 2019-10-30 22:00:01
@FilePath: \\机器学习实战\\05-Logistic回归\\logRegres.py
@Descripttion: Logistic回归主要是根据现有数据对分类边界线建立回归公式,进行分类
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def loadDataSet():
    """
    [summary]:加载数据
    
    Returns:
            dataMat -- 数据列表
            labelMat -- 标签列表
    """
    dataMat = []  # 创建数据列表
    labelMat = []  # 创建标签列表
    test_path = r'05-Logistic回归\testSet.txt'
    fp = open(test_path)
    for line in fp.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # 添加数据
        labelMat.append(int(lineArr[2]))  # 添加标签
    fp.close()  # 关闭文件
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


def Gradient_Ascent_test():
    """
    [summary]:梯度上升算法测试函数
              求f(x) = -x^2+4x的极大值
    Returns:
        x_new -- 函数极大值
    """
    def f_prime(x_old):  # f(x)的导师
        return -2 * x_old + 4

    x_old = -1
    x_new = 0
    alpha = 0.01
    pression = 0.00000001
    while abs(x_new - x_old) > pression:
        x_old = x_new
        x_new = x_old + alpha * f_prime(x_old)  # alpha控制幅度,f_prime控制方向
    print(x_new)


def gradAscent(dataMatIn, classLabels):
    """
    [summary]:梯度上升
    每个回归系数初始化为1
    重复R次:
        计算整个数据集的梯度
        使用alpha × gradient更新回归系数的向量
    返回回归系数
    
    Arguments:
        dataMatIn {[type]} -- 数据集
        classLabels {[type]} -- 数据标签
    
    Returns:
        weights.getA() -- # 将矩阵转换为数组，返回权重数组
        weights_array -- 每次更新的回归系数
    """
    dataMatrix = np.mat(dataMatIn)  # 转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()  # 转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)  # m为行数,n为列数。
    alpha = 0.001  # 步长
    maxCycles = 500  # 迭代次数
    weights = np.ones((n, 1))
    weights_array = np.array([])  # 每次更新的回归系数
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # 矩阵乘法
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
        weights_array = np.append(weights_array, weights)
    weights_array = weights_array.reshape(maxCycles, n)
    return weights.getA(), weights_array


def plotBestFit(weights):
    """
    [summary]:画出决策边界
    
    Arguments:
        wei {[type]} -- [description]
    """
    # weights = weights.getA()  # 将矩阵类型转化为数组
    dataMat, labelMat = loadDataSet()  # 加载数据集
    dataArr = np.array(dataMat)  # 转换成numpy的array数组
    n = np.shape(dataArr)[0]  # 数据个数
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    # print("dataArr:", dataArr)
    for i in range(n):
        if int(labelMat[i]) == 1:  # 1为正样本
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:  # 0为负样本
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 添加subplot
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')  # 绘制正样本
    ax.scatter(xcord2, ycord2, s=30, c='green')  # 绘制负样本
    x = np.arange(-3.0, 3.0, 0.1)
    Y = (-weights[0] - weights[1] * x) / weights[2]  # 0=w0X0+w1X1+w2X2 注意x0=1
    ax.plot(x, Y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stocGradAscent0(dataMatrix, classLabels):
    """
    [summary]:随机梯度上升
    所有回归系数初始化为1
    对数据集中每个样本
            计算该样本的梯度
            使用alpha × gradient更新回归系数值
            返回回归系数值
    
    Arguments:
        dataMatrix {[type]} -- [description]
        classLabels {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """
    [summary]:随机梯度上升算法
    
    Arguments:
        dataMatrix {[type]} -- 迭代数组
        classLabels {[type]} -- 数据标签
    
    Keyword Arguments:
        numIter {int} -- 最大迭代次数 (default: {150})
    
    Returns:
        weights -- 权重数组
        weights_array -- 每次更新的回归系数
    """
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)  # 初始化为1
    weights_array = np.array([])
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01  # 降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] *
                            weights))  # 选择随机选取的一个样本，每次迭代不使用已经使用过的样本点
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]  # 更新回归系数
            weights_array = np.append(weights_array, weights,
                                      axis=0)  # 添加回归系数到数组中
            del (dataIndex[randIndex])
    weights_array = weights_array.reshape(numIter * m, n)  # 改变维度
    return weights, weights_array


def plotWeights(weights_array1, weights_array2):
    """
    [summary]:绘制回归系数与迭代次数的关系
    
    Arguments:
        weights_array1 {[type]} -- 随机梯度上升算法每次更新的回归系数
        weights_array2 {[type]} -- 上升算法每次更新的回归系数
    """
    # 设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=3,nclos=2时,代表fig画布被分为六个区域,axs[0][0]表示第一行第一列
    fig, axs = plt.subplots(nrows=3,
                            ncols=2,
                            sharex=False,
                            sharey=False,
                            figsize=(20, 10))
    x1 = np.arange(0, len(weights_array1), 1)
    # 绘制w0与迭代次数的关系
    axs[0][0].plot(x1, weights_array1[:, 0])
    axs0_title_text = axs[0][0].set_title(u'随机梯度上升算法：回归系数与迭代次数关系',
                                          FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0', FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][0].plot(x1, weights_array1[:, 1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][0].plot(x1, weights_array1[:, 2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W2', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    x2 = np.arange(0, len(weights_array2), 1)
    # 绘制w0与迭代次数的关系
    axs[0][1].plot(x2, weights_array2[:, 0])
    axs0_title_text = axs[0][1].set_title(u'梯度上升算法：回归系数与迭代次数关系',
                                          FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0', FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][1].plot(x2, weights_array2[:, 1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][1].plot(x2, weights_array2[:, 2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W2', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    plt.show()


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    return 1.0 if prob > 0.5 else 0.0


def colicTest():
    frTest = open(r'05-Logistic回归\horseColicTest.txt', 'r', encoding='utf-8')
    frTrain = open(r'05-Logistic回归\horseColicTraining.txt',
                   'r',
                   encoding='utf-8')
    featuresNumbers = 21
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []

        for i in range(featuresNumbers):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[featuresNumbers]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels,
                                   1000)  # 使用改进的随即上升梯度训练
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(featuresNumbers):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(
                currLine[featuresNumbers]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec) * 100
    print("这个测试集的错误率: %.2f%%" % errorRate)
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("在 %d 迭代后,平均错误率: %f" % (numTests, errorSum / float(numTests)))


# if __name__ == '__main__':
#     dataMat, labelMat = loadDataSet()
#     weights1, weights_array1 = stocGradAscent1(np.array(dataMat), labelMat)

#     weights2, weights_array2 = gradAscent(dataMat, labelMat)
#     plotWeights(weights_array1, weights_array2)

from sklearn.linear_model import LogisticRegression


def colicSklearn():
    frTest = open(r'05-Logistic回归\horseColicTest.txt', 'r', encoding='utf-8')
    frTrain = open(r'05-Logistic回归\horseColicTraining.txt',
                   'r',
                   encoding='utf-8')
    trainingSet = []
    trainingLabels = []
    testSet = []
    testLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))
    classifier = LogisticRegression(solver='liblinear',
                                    max_iter=10).fit(trainingSet,
                                                     trainingLabels)
    test_accurcy = classifier.score(testSet, testLabels) * 100
    print('正确率:%f%%' % test_accurcy)


if __name__ == '__main__':
    colicSklearn()
