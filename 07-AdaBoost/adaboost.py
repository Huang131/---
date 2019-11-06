'''
@version: 0.0.1
@Author: Huang
@dev: python3 vscode
@Date: 2019-11-05 21:07:47
@LastEditTime: 2019-11-06 19:35:46
@FilePath: \\机器学习实战\\07-AdaBoost\\adaboost.py
@Descripttion: AdaBoost的大部分时间都用在训练上，分类器将多次在同一数据集上训练 弱分类器
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def loadSimpData():
    """
    [summary]:创建单层决策树的数据集
    
    Returns:
       datMat -- np.matrix
       classLabels -- List
    """
    # 数据矩阵
    datMat = np.matrix([[1., 2.1], [1.6, 1.4], [1.3, 1.], [1., 1.], [2., 1.]])
    # 数据标签
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def showDataSet(dataMat, labelMat):
    """
    数据可视化
    Parameters:
        dataMat - 数据矩阵
        labelMat - 数据标签
    Returns:
        无
    """
    data_plus = []  # 正样本
    data_minus = []  # 负样本

    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])

    data_plus_np = np.array(data_plus)  # 转换为numpy矩阵
    data_minus_np = np.array(data_minus)  # 转换为numpy矩阵
    # np.transpose 将X和Y分别置于同行
    plt.scatter(np.transpose(data_plus_np)[0],
                np.transpose(data_plus_np)[1])  # 正样本散点图
    plt.scatter(
        np.transpose(data_minus_np)[0],
        np.transpose(data_minus_np)[1])  # 负样本散点图
    plt.show()


def loadDataSet(fileName):
    """
    [summary]:加载数据集
    
    Arguments:
        fileName  -- 文件名
    
    Returns:
        dataMat  -- 数据
        labelMat --标签
    """
    # 字段数目
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    [summary]:单层决策树分类函数,根据某一特征进行分类
    
    Arguments:
        dataMatrix  -- 数据矩阵
        dimen -- 选取第几列,对特征进行抽取
        threshVal  -- 阀值
        threshIneq  -- 比较关系(lt)
    
    Returns:
        retArray [numpy.ndarray]-- 分类结果
    """
    # 初始化retArray为1
    retArray = np.ones((np.shape(dataMatrix)[0], 1))

    # 置于-1,进行分类
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0

    return retArray


def buildStump(dataArr, classLabels, D):
    """
    [summary]:找到数据集上最佳的单层决策树
    将最小错误率minError设为+∞   
    对数据集中的每一个特征（第一层循环）:
            对每个步长（第二层循环）:
            对每个不等号（第三层循环）:
                    建立一棵单层决策树并利用加权数据集对它进行测试
                    如果错误率低于minError，则将当前单层决策树设为最佳单层决策树
    返回最佳单层决策树
    
    Arguments:
        dataArr  -- 数据矩阵
        classLabels  -- 数据标签
        D -- 样本权重
    
    Returns:
        bestStump - 最佳单层决策树信息
        minError - 最小误差
        bestClasEst - 最佳的分类结果
    """
    # 初始化操作
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = np.inf  # 最小误差初始化为正无穷大

    for i in range(n):  # 遍历所有特征
        # 找到特征中最小的值和最大值
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps  # 步长
        # 在当前特征上进行遍历
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:  # 大于和小于的情况
                threshVal = (rangeMin + float(j) * stepSize)  # 计算阈值
                # 计算分类结果
                predictedVals = stumpClassify(dataMatrix, i, threshVal,
                                              inequal)
                # 初始化误差矩阵
                errArr = np.mat(np.ones((m, 1)))
                # 分类正确的,赋值为0
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr  # 计算误差(1,5) (5,1)
                print(
                    "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f"
                    % (i, threshVal, inequal, weightedError))
                # 找到误差最小的分类方式
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
    [summary]:
    对每次迭代：
        利用buildStump()函数找到最佳的单层决策树
        将最佳单层决策树加入到单层决策树数组
        计算alpha
        计算新的权重向量D
        更新累计类别估计值
        如果错误率等于0.0，则退出循环
    
    Arguments:
        dataArr {[type]} -- 数据
        classLabels {[type]} -- 标签
    
    Keyword Arguments:
        numIt {int} -- 迭代次数 (default: {40})
    
    Returns:
        weakClassArr
        aggClassEst
    """
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)  # 概率分布向量,元素之和为1。D在迭代中增加错分数据的权重
    aggClassEst = np.mat(np.zeros((m, 1)))  # 记录每个数据点的类别估计累计值

    for i in range(numIt):
        # 构建单层决策树
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D:", D.T)
        # 根据公式计算弱学习算法权重alpha,使error不等于0,因为分母不能为0
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha  # 存储弱学习算法权重
        weakClassArr.append(bestStump)  # 存储单层决策树
        # print("classEst: ", classEst.T)
        # 根据数学公式更改权重
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        # 计算AdaBoost误差，当误差为0的时候，退出循环
        aggClassEst += alpha * classEst
        print("aggClassEst: ", aggClassEst.T)
        aggErrors = np.multiply(
            np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate)
        if errorRate == 0.0:
            break
    # print('weakClassArr:', weakClassArr, "aggClassEst", aggClassEst)
    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    """
    [summary]:AdaBoost分类函数
    
    Arguments:
        datToClass  -- 待分类样例
        classifierArr -- 训练好的分类器
    
    Returns:
         分类结果
    """
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))

    # 遍历所有分类器，进行分类
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)


def plotROC(predStrengths, classLabels):
    """
    [summary]:绘制ROC
    
    Arguments:
        predStrengths -- 分类器的预测强度
        classLabels -- 类别
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    cur = (1.0, 1.0)  # 绘制光标的位置
    ySum = 0.0  # 用于计算AUC
    numPosClas = np.sum(np.array(classLabels) == 1.0)  # 统计正类的数量
    yStep = 1 / float(numPosClas)  # y轴步长
    xStep = 1 / float(len(classLabels) - numPosClas)  # x轴步长
    sortedIndicies = predStrengths.argsort()  # get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]  # 高度累加
        # 绘制ROC
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        # 更新绘制光标的位置
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('假阳率 FP', FontProperties=font)
    plt.ylabel('真阳率 TP', FontProperties=font)
    plt.title('AdaBoost马疝病检测系统的ROC曲线', FontProperties=font)
    ax.axis([0, 1, 0, 1])  # 设置刻度
    plt.show()
    print("AUC面积为:", ySum * xStep)
    plt.show()


# 自己写的daBoostTrainDS实现病马分类
def main():
    dataArr, LabelArr = loadDataSet(r'.\07-AdaBoost\horseColicTraining2.txt')
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, LabelArr)
    testArr, testLabelArr = loadDataSet(r'.\07-AdaBoost\horseColicTest2.txt')
    predictions = adaClassify(dataArr, weakClassArr)
    errArr = np.mat(np.ones((len(dataArr), 1)))
    print('训练集的错误率:%.3f%%' % float(
        errArr[predictions != np.mat(LabelArr).T].sum() / len(dataArr) * 100))
    predictions = adaClassify(testArr, weakClassArr)
    errArr = np.mat(np.ones((len(testArr), 1)))
    print('测试集的错误率:%.3f%%' %
          float(errArr[predictions != np.mat(testLabelArr).T].sum() /
                len(testArr) * 100))


def main_sklearn():
    # 用sklearn实现病马分类
    dataArr, LabelArr = loadDataSet(r'.\07-AdaBoost\horseColicTraining2.txt')
    testArr, testLabelArr = loadDataSet(r'.\07-AdaBoost\horseColicTest2.txt')

    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                             algorithm='SAMME',
                             n_estimators=10)
    bdt.fit(dataArr, LabelArr)
    predictions = bdt.predict(dataArr)
    errArr = np.mat(np.ones((len(dataArr), 1)))
    print('训练集的错误率:%.3f%%' %
          float(errArr[predictions != LabelArr].sum() / len(dataArr) * 100))
    predictions = bdt.predict(testArr)
    errArr = np.mat(np.ones((len(testArr), 1)))
    print(
        '测试集的错误率:%.3f%%' %
        float(errArr[predictions != testLabelArr].sum() / len(testArr) * 100))


def main_roc():
    dataArr, LabelArr = loadDataSet(r'.\07-AdaBoost\horseColicTraining2.txt')
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, LabelArr)
    plotROC(aggClassEst.T, LabelArr)


if __name__ == '__main__':
    main_roc()
