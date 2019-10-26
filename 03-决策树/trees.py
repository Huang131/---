"""
@Descripttion: 决策树
优点:计算复杂度不高,输出结果易于理解,对中间值的缺失不敏感,可以处理不相关特征数据
缺点:可能会产生过度匹配问题
适用数据类型:数值型和标称型
@version: 0.0.1
@Author: Huang
@dev: python3 vscode
@Date: 2019-06-03 12:15:20
@LastEditors: Huang
@LastEditTime: 2019-10-26 02:17:15
"""

from math import log
import operator
import pickle


def createDataSet():
    """
    [summary]:创建数据集
    
    Returns:
            dataSet - 数据集
            labels - 分类属性
    """
    dataSet = [
        [0, 0, 0, 0, "no"],
        [0, 0, 0, 1, "no"],
        [0, 1, 0, 1, "yes"],
        [0, 1, 1, 0, "yes"],
        [0, 0, 0, 0, "no"],
        [1, 0, 0, 0, "no"],
        [1, 0, 0, 1, "no"],
        [1, 1, 1, 1, "yes"],
        [1, 0, 1, 2, "yes"],
        [1, 0, 1, 2, "yes"],
        [2, 0, 1, 2, "yes"],
        [2, 0, 1, 1, "yes"],
        [2, 1, 0, 1, "yes"],
        [2, 1, 0, 2, "yes"],
        [2, 0, 0, 0, "no"],
    ]
    labels = ["年龄", "有工作", "有自己的房子", "信贷情况"]
    # 返回数据集和分类属性
    return dataSet, labels


def calcShannonEnt(dataSet):
    """
    [summary]:计算给定数据集的香农熵
    
    Arguments:
        dataSet  -- 数据集
    
    Returns:
        shannonEnt - 经验熵(香农熵)
    """
    # 返回数据集的行数
    numEntries = len(dataSet)
    labelCounts = {}  # 为所有可能的分类创建字典
    for featVec in dataSet:  # 对每组特征向量进行统计
        currentLabel = featVec[-1]  # 提取标签(Label)信息
        # Label计数
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1

    shannonEnt = 0.0  # 经验熵(香农熵)
    for key in labelCounts:
        prob = float(labelCounts[key] / numEntries)
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    """
    [summary]: 按照给定特征划分数据集
    
    Arguments:
        dataSet {[type]} -- 待划分的数据集
        axis {[type]} -- 划分数据集的特征
        value {[type]} -- 需要返回的特征的值
    
    Returns:
        retDataSet -- 返回的数据集列表
    """
    retDataSet = []
    for featVec in dataSet:  # 遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis] + featVec[axis + 1 :]
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    [summary]:选择最好的数据集划分方式(信息增益最大)
    
    Arguments:
        dataSet {[type]} -- 数据集
    
    Returns:
        bestFeature -- 最优特征特征的索引值
    """
    numFeatures = len(dataSet[0]) - 1  # 特征数量
    baseENtropy = calcShannonEnt(dataSet)  # 整个数据集的原始香农熵
    bestInfoGain = 0.0  # 信息增益
    bestFeature = -1  # 最优特征的索引值
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 创建唯一的分类标签,set元素不可重复
        newEntropy = 0.0  # 经验条件熵
        # 计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)  # subDataSet划分后的子集
            prob = len(subDataSet) / float(len(dataSet))  # 计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseENtropy - newEntropy  # 信息增益
        print("第%d个特征的增益为%.3f" % (i, infoGain))
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    """
    [summary]:统计classList中出现次数最多的元素(类标签)
    
    Arguments:
        classList {[type]} -- 类标签列表
    
    Returns:
        sortedClassCount[0][0] -- 出现此处最多的元素(类标签)
    """

    classCount = {}
    for vote in classList:  # 统计classList中每个元素出现的次数
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True
    )  # 根据字典的值降序排序
    return sortedClassCount[0][0]  # 返回classList中出现次数最多的元素


def createTree(dataSet, labels, featLabels):
    """
    [summary]:创建决策树
    
    Arguments:
        dataSet -- 训练数据集
        labels -- 分类属性标签
        featLabels -- 存储选择的最优特征标签
    
    Returns:
         myTree - 决策树
    """
    classList = [example[-1] for example in dataSet]  # 取分类标签(是否放贷:yes or no)
    if classList.count(classList[0]) == len(classList):  # 如果类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:  # 遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择最优特征
    bestFeatLabel = labels[bestFeat]  # 最优特征的标签
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel: {}}  # 根据最优特征的标签生成树
    del labels[bestFeat]  # 删除已经使用特征标签
    featValues = [example[bestFeat] for example in dataSet]  # 得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)  # 去掉重复的属性值
    for value in uniqueVals:  # 遍历特征，创建决策树。
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value), labels, featLabels
        )
    return myTree


def classify(inputTree, featLabels, testVec):
    """
    [summary]:使用决策树分类
    
    Arguments:
        inputTree -- 已经生成的决策树
        featLabels -- 存储选择的最优特征标签
        testVec -- 测试数据列表，顺序对应最优特征标签
    
    Returns:
         classLabel - 分类结果
    """
    firstStr = next(iter(inputTree))  # 获取决策树结点
    secondDict = inputTree[firstStr]  # 下一个字典
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == "dict":
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    """
    [summary]:存储决策树
    
    Arguments:
        inputTree {[type]} -- 已经生成的决策树
        filename {[type]} -- 决策树的存储文件名
    """
    with open(filename, "wb") as fw:
        pickle.dump(inputTree, fw)


def grabTree(filename):
    """
    [summary]:读取决策树
    
    Arguments:
        filename {[type]} -- 决策树的存储文件名
    
    Returns:
        pickle.load(fr) -- 决策树字典
    """
    fr = open(filename, "rb")
    return pickle.load(fr)


if __name__ == "__main__":
    dataSet, labels = createDataSet()
    featLabels = []
    myTree = createTree(dataSet, labels, featLabels)
    testVec = [0, 1]  # 测试数据
    result = classify(myTree, featLabels, testVec)
    if result == "yes":
        print("放贷")
    if result == "no":
        print("不放贷")

