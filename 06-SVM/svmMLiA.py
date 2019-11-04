'''
@version: 0.0.1
@Author: Huang
@dev: python3 vscode
@Date: 2019-10-31 00:05:29
@LastEditTime: 2019-11-04 23:15:13
@FilePath: \\机器学习实战\\06-SVM\\svmMLiA.py
@Descripttion: SVM的大部分时间都源自训练，该过程主要实现两个参数的调优
'''

import numpy as np


def loadDataSet(fileName):
    """
    [summary]:加载数据
    
    Arguments:
        fileName {[str]} -- 文件路径
    
    Returns:
        [type] -- [description]
    """
    dataMat = []
    labelMat = []
    fp = open(fileName)
    for line in fp.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])  # 添加数据
        labelMat.append(float(lineArr[2]))  # 添加标签
    return dataMat, labelMat


def selectJrand(i, m):
    """
    [summary]:随机选择alpha
    
    Arguments:
        i -- alpha
        m -- aplaha参数个数
    Returns:
        j
    """
    j = i
    while j == i:
        j = int(np.random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    """
    [summary]:修建alpha
    
    Arguments:
        aj -- alpha值
        H  -- alpha上限
        L  -- alpha下限
    
    Returns:
        aj -- alpha值
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """
    [summary]:简化版SMO算法
    创建一个alpha向量并将其初始化为0向量
    当迭代次数小于最大迭代次数时（外循环)
        对数据集中的每个数据向量(内循环)
            如果该数据向量可以被优化:
            随机选择另外一个数据向量
            同时优化这两个向量
            如果两个向量都不能被优化，退出内循环
        如果所有向量都没被优化，增加迭代数目，继续下一次循环
    Arguments:
        dataMatIn {[type]} -- 数据矩阵
        classLabels {[type]} -- 数据标签
        C {[type]} -- 松弛变量
        toler {[type]} -- 容错率
        maxIter {[type]} -- 最大迭代次数
    """
    # 将参数转化为Numpy矩阵,简化数学操作
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()  # 转置成列向量
    b = 0  # *初始化b参数
    m, n = np.shape(dataMatrix)  # 行 列
    alphas = np.mat(np.zeros((m, 1)))  # *初始化alph参数为0
    iter = 0  # 没有alpha改变的情况下,遍历数据集的次数
    while (iter < maxIter):
        alphaPairsChanged = 0  # *记录alpha是否已经进行优化
        for i in range(m):
            # 预测类别
            fXi = float(
                np.multiply(alphas, labelMat).T *
                (dataMatrix * dataMatrix[i, :].T)
            ) + b  # if checks if an example violates KKT conditions

            # *计算误差Ei
            Ei = fXi - float(
                labelMat[i])  # if checks if an example violates KKT conditions
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or\
                             ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):

                j = selectJrand(i, m)  # 随机选择第二个alpha
                # *计算误差Ej
                fXj = float(
                    np.multiply(alphas, labelMat).T *
                    (dataMatrix * dataMatrix[j, :].T)) + b

                Ej = fXj - float(labelMat[j])
                # *保存更新前的alpha值
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # *计算上下界L和H
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue

                # *Eta是alpha[j]的最优修改量
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[
                    i, :] * dataMatrix[i, :].T - dataMatrix[j, :] * dataMatrix[
                        j, :].T

                if eta >= 0:
                    print("eta>=0")
                    continue

                # 更新alphas[j]
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("alpha[j]变化不够大")
                    continue
                # 更新alphas[i],b1,b2
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold -
                                                          alphas[j])
                b1 = b - Ei - labelMat[i] * (
                    alphas[i] - alphaIold
                ) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMat[j] * (
                    alphas[j] -
                    alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (
                    alphas[i] - alphaIold
                ) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[j] * (
                    alphas[j] -
                    alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T

                # 更新b
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1  # 统计优化次数
                print("第%d次迭代 样本:%d, alpha优化次数:%d" %
                      (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("迭代次数: %d" % iter)
    return b, alphas


def kernelTrans(X, A, kTup):
    """
    [summary]: 通过核函数将数据转换更高维的空间
    
    Arguments:
        X -- 数据矩阵
        A -- 单个数据的向量
        kTup -- 包含核函数信息的元组
    
    Raises:
        NameError: 核函数名称错误
    
    Returns:
         K - 计算的核K
    """
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T  # 线性核函数,只进行内积
    elif kTup[0] == 'rbf':  # 高斯核函数,根据高斯核函数公式进行计算
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1]**2))  # 计算高斯核K
    else:
        raise NameError('核函数名称错误,无法识别')
    return K


class optStruct:
    """
    [summary]:维护所有需要操作的值
    Parameters：
        dataMatIn - 数据矩阵
        classLabels - 数据标签
        C - 松弛变量
        toler - 容错率
    """
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn  # 数据矩阵
        self.labelMat = classLabels  # 数据标签
        self.C = C  # 松弛变量
        self.tol = toler  # 容错率
        self.m = np.shape(dataMatIn)[0]  # 数据矩阵行数
        self.alphas = np.mat(np.zeros((self.m, 1)))  # 根据矩阵行数初始化alpha参数为0
        self.b = 0  # 初始化b参数为0
        self.eCache = np.mat(np.zeros(
            (self.m, 2)))  # 根据矩阵行数初始化虎误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def calcEk(oS, k):
    """
    [summary]:计算误差
    
    Arguments:
        oS  -- 传递的结构
        k   -- 标记为k的数据
    
    Returns:
        Ek - 标号为k的数据误差
    """
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):
    """
    [summary]:内循环启发方式
    
    Arguments:
        i  -- 标号为i的数据的索引值
        oS -- 数据结构
        Ei -- 标号为i的数据误差
    
    Returns:
        j, maxK - 标号为j或maxK的数据的索引值
        Ej - 标号为j的数据误差
    """
    # 初始化相关参数
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]  # 根据Ei更新误差缓存
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]  # 返回非零E值所对应的alpha值
    if (len(validEcacheList)) > 1:  # 有不为0的误差
        for k in validEcacheList:
            if k == i:
                continue  # 不计算i,浪费时间
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):  # 选择具有最大步长的j
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej  # 返回maxK,Ej
    else:  # 没有不为0的误差
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


# 计算误差值并存入缓存
def updateEk(oS, k):
    """
    [summary]:计算Ek,并更新误差缓存
    
    Arguments:
        oS -- 数据结构
        k  -- 标号为k的数据的索引值
    """
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]  #更新误差缓存


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    """
    [summary]:完整的线性SMO算法
    
    Arguments:
       dataMatIn -- 数据矩阵
        classLabels -- 数据标签
        C -- 松弛变量
        toler -- 容错率
        maxIter -- 最大迭代次数
    
    Keyword Arguments:
        kTup {tuple} -- [description] (default: {('lin', 0)})
    
    Returns:
        oS.b - SMO算法计算的b
        oS.alphas - SMO算法计算的alphas
    """
    # 初始化数据结构
    oS = optStruct(np.mat(dataMatIn),
                   np.mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    # 遍历整个数据集都alpha也没有更新或者超过最大迭代次数,则退出循环
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:  # go over all
            for i in range(oS.m):
                alphaPairsChanged += np.innerL(i, oS)
                print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" %
                      (iter, i, alphaPairsChanged))
            iter += 1
        else:  # 遍历不在边界0和C的alpha
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += np.innerL(i, oS)
                print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" %
                      (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False  # 如果alpha没有更新,计算全样本遍历
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("迭代次数: %d" % iter)
    return oS.b, oS.alphas


if __name__ == '__main__':
    dataArr, labelArr = loadDataSet(r'06-SVM\testSet.txt')
    b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    print(b, alphas[alphas > 0])
