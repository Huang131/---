'''
@version: 0.0.1
@Author: Huang
@dev: python3 vscode
@Date: 2019-11-12 21:09:57
@LastEditTime: 2019-11-12 21:58:07
@FilePath: \\机器学习实战\\14-SVD算法\\SVD.py
@Descripttion: SVD是从有噪声数据中抽取相关特征,利用SVD来逼近矩阵并从中提取重要特征,通过保留矩阵80%~90%的能量,
就可以得到重要的特征并去掉噪声
'''


import numpy as np


def loadExData():
    return [[0, 0, 0, 2, 2],
            [0, 5, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 1, 2, 4, 0],
            [5, 5, 5, 3, 0],
            [1, 1, 1, 2, 0]]


def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


def ecludSim(inA, inB):
    """计算两个列向量的欧氏距离"""
    return 1.0 / (1.0 + np.linalg.norm(inA - inB))


def pearsSim(inA, inB):
    """计算两个列向量的皮尔逊相关系数"""
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=False)[0][1]


def cosSim(inA, inB):
    """计算两个列向量的是余弦相似度"""
    num = float(inA.T * inB)
    denom = np.linalg.norm(inA) * np.linalg.norm(inB)
    return 0.5 + 0.5 * (num / denom)


def test_Sim():
    """测试三种距离算法"""
    myDat = np.mat(loadExData())
    ecl = ecludSim(myDat[:, 0], myDat[:, 4])
    print(ecl)
    cos = cosSim(myDat[:, 0], myDat[:, 4])
    print(cos)
    pear = pearsSim(myDat[:, 0], myDat[:, 4])
    print(pear)


def standEst(dataMat, user, simMeas, item):
    """
    [summary]:计算在给定相似度计算方法的 条件下，用户对物品的估计评分值

    Arguments:
        dataMat  -- 数据集
        user -- 
        simMeas  -- 
        item -- 
    
    Returns:
        [type] -- [description]
    """
    n = np.shape(dataMat)[1] # 行数
    simTotal = 0.0
    ratSimTotal = 0.0

    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        # 寻找两个用户都评级的物品
        overlap = np.nonzero(
            np.logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        # 没有任何重合元素,相似度为0
        if len(overlap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overlap, j], dataMat[overlap, item])
        print('the {:d} and {:d} similarity is:{:.6f}'.format(
            item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    """
    [summary]:
        (1) 寻找用户没有评级的菜肴，即在用户－物品矩阵中的0值
        (2) 在用户没有评级的所有物品中，对每个物品预计一个可能的评级分数。这就是说，我们认为用户可能会对物品的打分(这就是相似度计算的初衷)
        (3) 对这些物品的评分从高到低进行排序，返回前N个物品
    
    Arguments:
        dataMat {[type]} -- [description]
        user {[type]} -- [description]
    
    Keyword Arguments:
        N {int} -- [description] (default: {3})
        simMeas {[type]} -- [description] (default: {cosSim})
        estMethod {[type]} -- [description] (default: {standEst})
    
    Returns:
        [type] -- [description]
    """
    # 寻找未评级物品
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0:
        return 'all rated'
    itemScores = [(item, estMethod(dataMat, user, simMeas, item))
                  for item in unratedItems]
    # itemScores = []
    # for item in unratedItems:
    #     estimatedScore=estMethod(dataMat,user,simMeas,item)
    #     itemScores.append((item,estimatedScore))
    # 寻找前N个未评级物品
    return sorted(itemScores, key=lambda x: x[1], reverse=True)[:N]


def test_recommend():
    myDat = np.mat(loadExData())
    myDat[0, 1] = myDat[0, 0] = myDat[1, 0] = myDat[2, 0] = 4
    myDat[3, 3] = 2
    print(recommend(myDat, 2))


def svdEst(dataMat, user, simMeas, item):
    """
    [summary]:数对给定用户给定物品构建了一个评分估计值
    
    Arguments:
        dataMat {[type]} -- [description]
        user {[type]} -- [description]
        simMeas {[type]} -- [description]
        item {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    # 行数
    n = np.shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    # SVD分解
    U, Sigma, VT = np.linalg.svd(dataMat)
    # 构建对角矩阵,Sigma[:4]只包含90%能量值的奇异值
    Sig4 = np.mat(np.eye(4) * Sigma[:4])
    xfromedItems = dataMat.T * U[:, :4] * Sig4.I
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeas(xfromedItems[item, :].T, xfromedItems[j, :].T)
        print('the {:d} and {:d} similarity is:{:.6f}'.format(
            item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def test_svdEst():
    myDat = np.mat(loadExData2())
    # print(recommend(myDat, 1, estMethod=svdEst))
    print(recommend(myDat, 1, estMethod=svdEst, simMeas=pearsSim))


# 打印矩阵
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1, end='')  # 不换行
            else:
                print(0, end='')
        print(' ')


def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open(r'14-SVD算法\0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = np.mat(myl)
    print("****original matrix******")
    printMat(myMat, thresh)
    U, Sigma, VT = np.linalg.svd(myMat)
    SigRecon = np.mat(np.zeros((numSV, numSV)))
    for k in range(numSV):  # construct diagonal matrix from vector
        SigRecon[k, k] = Sigma[k]
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)


if __name__ == '__main__':
    # test_Sim()
    # test_recommend()
    # test_svdEst()
    imgCompress(2)
