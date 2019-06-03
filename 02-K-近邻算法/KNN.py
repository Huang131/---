'''
Created on 2019-6-1
kNN: k Nearest Neighbors

优点:分类数据最简单最有效的算法

缺点:无法给出任何数据的基础结构信息,无法知晓平均实例样本和典型实例样本具有的特诊

@author: huang
'''
from numpy import *
import operator
from os import listdir


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    '''
    inX:用于分类的向量
    dataSet:输入的训练集
    labels:标签向量
    k:选择最近邻的数目
    '''
    # 距离计算
    dataSetSize = dataSet.shape[0]  # 训练集行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # (维度倍数,重复次数)
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5  # 欧式距离
    sortedDistIndicies = distances.argsort()  # 返回数组值从小到大的索引值
    classCount = {}  # {'B': 2, 'A': 1}
    # 选择距离最小的K个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    '''
    将文本记录到转换NumPy的解析程序
    '''
    fr = open(filename)
    arrayLines = fr.readlines()
    numberOfLines = len(arrayLines)  # 得到文件行数
    returnMat = zeros((numberOfLines, 3))  # 创建返回NumPy矩阵
    classLabelVector = []
    index = 0
    # 解析文件数据到文本
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    '''
    归一化
    newValue = (oldValue - min)/(max-min)
    '''
    minVals = dataSet.min(0)  # 每列最小值
    maxVals = dataSet.max(0)  # 每列最大值
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))  # 特征值相除
    return normDataSet, ranges, minVals


def datingClassTest():
    '''
    分类器针对约会网站的测试代码
    '''
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]  # 行数
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m],
                                     3)  # 向量 训练集 标签 k
        print("the classifier came back with: %d,the real answer is :%d" %
              (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
        print("the total error rate is:%f" % (errorCount / float(numTestVecs)))


def classifyPerson():
    '''
    预测对某人的喜欢程度
    '''
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat,
                                 datingLabels, 3)
    print("you will probably like this person:",
          resultList[classifierResult - 1])


def img2vector(filename):
    '''
    将32 * 32图像转换为1 * 1024向量
    '''
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    '''
    手写数字识别系统的测试代码
    '''
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' %
                                       fileNameStr)
    testFileList = listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("分类结果为:%d,实际结果为:%d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("错误次数: %d" % errorCount)
    print("错误率: %f" % (errorCount / float(mTest)))
