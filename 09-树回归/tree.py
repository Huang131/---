'''
@version: 0.0.1
@Author: Huang
@dev: python3 vscode
@Date: 2019-11-07 23:59:30
@LastEditTime: 2019-11-08 14:16:04
@FilePath: \\机器学习实战\\09-树回归\\tree.py
@Descripttion: CART是十分著名且广泛记载的树构建算法，它使用二元切分来处理连续型变量
'''

import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    """
    [summary]:加载数据
    """
    dataMat = []
    fp = open(fileName)
    for line in fp.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


def plotDataSet(fileName):
    """
    [summary]:可视化数据集
    
    Arguments:
        fileName {[str]} -- 文件名
    """
    dataMat = loadDataSet(fileName)
    n = len(dataMat)
    xcord, ycord = [], []
    for i in range(n):
        xcord.append(dataMat[i][0])
        ycord.append(dataMat[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord, ycord, s=20, c='blue', alpha=0.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def binSplitDataSet(dataSet, feature, value):
    """
    [summary]:切分数据集合
    
    Arguments:
        dataSet {[type]} -- 数据集合
        feature {[type]} -- 待切分特征
        value {[type]} -- 阈值
    
    Returns:
        mat0 -- 大于特征的切分子集0
        mat1 -- 小于等于特征的切分子集1
    """
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    """
    [summary]:生成叶节点
    
    Arguments:
        dataSet  -- 数据集合
    
    Returns:
        目标变量的均值
    """
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    """
    [summary]:计算总方差
    """
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def linearSolve(dataSet):
    """
    [summary]:将数据集格式化成目标变量Y和自变量X
    
    Arguments:
        dataSet {[type]} -- [description]
    
    Raises:
        NameError: [description]
    
    Returns:
        [type] -- [description]
    """
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))  # create a copy of data with 1 in 0th postion
    X[:, 1:n] = dataSet[:, 0:n - 1]
    Y = dataSet[:, -1]  # and strip out Y
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def modelLeaf(dataSet):
    """
    [summary]:create linear model and return coeficients
    """
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    """
    [summary]:计算误差
    """
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(np.power(Y - yHat, 2))


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    [summary]:找到数据的最佳二元切分方式函数
        对每个特征:
            对每个特征值:
                将数据集切分成两份
                计算切分的误差
                如果当前误差小于当前最小误差，那么将当前切分设定为最佳切分并更新最小误差
            返回最佳切分的特征和阈值
    
    Arguments:
        dataSet {[numpy.matrix]} -- 数据集合
    
    Keyword Arguments:
        leafType {[type]} -- 建立叶节点的函数 (default: {regLeaf})
        errType {[type]} -- 误差计算函数 (default: {regErr})
        ops {tuple} -- 包含树构建其他所需参数的元组 (default: {(1, 4)})        
    
    Returns:
        bestIndex -- 最佳切分特征
        bestValue -- 特征值
    """
    tolS = ops[0]  # 误差下降值
    tolN = ops[1]  # 切分的最少样本数
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)

    m, n = np.shape(dataSet)
    # 默认最后一个特征为最佳切分特征,计算其误差估计
    S = errType(dataSet)
    bestS = np.inf  # 最佳误差
    bestIndex = 0  # 特征切分索引
    bestValue = 0  # 特征值
    # 遍历所有特诊列
    for featIndex in range(n - 1):
        # 遍历所有特征值
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            # 根据特征和特征值切分数据集
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            # 计算误差估计
            newS = errType(mat0) + errType(mat1)
            # 如果误差估计更小,则更新特征索引值和特征值
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    # 根据最佳的切分特征和特征值切分数据集合
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 如果切分出的数据集很小则退出
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):  # exit cond 3
        return None, leafType(dataSet)
    # 返回最佳切分特征和特征值
    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    [summary]:
        找到最佳的待切分特征：
            如果该节点不能再分，将该节点存为叶节点
            执行二元切分
            在右子树调用createTree()方法
            在左子树调用createTree()方法
    
    Arguments:
        dataSet {[type]} -- 数据集
    
    Keyword Arguments:
        leafType {[type]} -- 建立叶节点的函数 (default: {regLeaf})
        errType {[type]} -- 误差计算函数 (default: {regErr})
        ops {tuple} -- 包含树构建其他所需参数的元组 (default: {(1, 4)})
    
    Returns:
        [type] -- [description]
    """
    # 选择最好的切分
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}  # 回归树
    retTree['spInd'] = feat
    retTree['spVal'] = val
    # 分成左数据集和右数据集
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def isTree(obj):
    """
    [summary]:判断当前处理节点是否是叶节点
    """
    return (type(obj).__name__ == 'dict')


def getMean(tree):
    """
    [summary]:塌陷处理,从上到下遍历树,计算找到的两个叶节点的平均值
    """
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):
    """
    [summary]:后剪枝
        基于已有的树切分测试数据:
        如果存在任一子集是一棵树，则在该子集递归剪枝过程
        计算将当前两个叶节点合并后的误差
        计算不合并的误差
        如果合并会降低误差的话，就将叶节点合并
    
    Arguments:
        tree {[type]} -- 待剪枝的树
        testData {[type]} -- 测试集
    
    Returns:
        [type] -- 树的平均值
    """
    if np.shape(testData)[0] == 0:
        return getMean(tree)  # if we have no test data collapse the tree
    # 如果有左子树或者右子树,则切分数据集
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    # 处理左子树(剪枝)
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    # 处理右子树(剪枝)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    # 如果当前结点的左右结点为叶结点
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # 计算没有合并的误差
        errorNoMerge = sum(np.power(lSet[:, -1] - tree['left'], 2)) +\
            sum(np.power(rSet[:, -1] - tree['right'], 2))
        # 计算合并的均值
        treeMean = (tree['left'] + tree['right']) / 2.0
        # 计算合并的误差
        errorMerge = sum(np.power(testData[:, -1] - treeMean, 2))
        # 如果合并的误差小于没有合并的误差,则合并
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat


if __name__ == '__main__':
    myDat = loadDataSet(r'./09-树回归/ex00.txt')
    myMat = np.mat(myDat)
    print(createTree(myMat))
