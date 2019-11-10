'''
@version: 0.0.1
@Author: Huang
@dev: python3 vscode
@Date: 2019-11-10 11:39:30
@LastEditTime: 2019-11-10 18:11:16
@FilePath: \\机器学习实战\\10-K均值聚类算法\\kMeans.py
@Descripttion: 聚类是一种无监督的学习，它将相似的对象归到同一个簇中
'''

import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(filename):
    """
    [summary]:加载数据
    
    Arguments:
        filename  -- 文件名
    
    Returns:
        [List] -- 数据集
    """
    dataMat = []
    with open(filename) as fr:
        for line in fr.readlines():
            curline = line.strip().split()
            fltline = list(map(float, curline))
            dataMat.append(fltline)
    return dataMat


def distEclud(vecA, vecB):
    """
    [summary]:计算两个向量的欧氏距离
    
    Arguments:
        vecA  -- A坐标
        vecB  -- B坐标
    
    Returns:
        两点之间的欧氏距离
    """
    # not sum(mat) but mat.sum()
    return np.sqrt(np.power(vecA - vecB, 2).sum())


def randCent(dataSet, k):
    """
    [summary]:为数据集构建k个随机质心的集合
    
    Arguments:
        dataSet {[mat]} -- 数据集
        k {[int} -- 聚类数
    
    Returns:
        [mat] -- k个中心点组成的矩阵
    """
    n = np.shape(dataSet)[1]  # 获取列数
    centroids = np.mat(np.zeros((k, n)))
    # 遍历所有列
    for j in range(n):
        minj = min(dataSet[:, j])
        rangej = float(max(dataSet[:, j]) - minj)
        # 最小值+区间×随机系数,确保生成的中心点在数据集边界之内
        centroids[:, j] = minj + rangej * np.random.rand(k, 1)
    return centroids


def my_KMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    """
    [summary]:
        创建k个点作为起始质心（经常是随机选择）
        当任意一个点的簇分配结果发生改变时
            对数据集中的每个数据点
                对每个质心
                    计算质心与数据点之间的距离
                将数据点分配到距其最近的簇
            对每一个簇，计算簇中所有点的均值并将均值作为质心
    
    Arguments:
        dataSet {[mat]} -- 数据集
        k {[int]} -- 聚类数
    
    Keyword Arguments:
        distMeas  -- 距离算法 (default: {distEclud})
        createCent -- 创建初始质心 (default: {randCent})
    
    Returns:
        centroids -- (k,n) 类质心
        clusterAssment -- (m,2) 点分配
    """
    m = np.shape(dataSet)[0]  # 行数
    # 簇分配结果：第一列记录索引值;第二列存储误差,当前点到簇质心的距离
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 随机生成中心点完成初始化
    centroids = randCent(dataSet, k)  # (k,n)
    clusterchanged = True
    while clusterchanged:
        # 假定所有点分配都不发生改变，标记为False
        clusterchanged = False
        for i in range(m):
            cluster_i = clusterAssment[i, 0]  # 取出簇索引值
            dismax = np.inf
            for j in range(k):
                curdis = distEclud(centroids[j, :], dataSet[i, :])
                if curdis < dismax:
                    dismax = curdis
                    # 更新簇分配结果
                    clusterAssment[i, :] = j, dismax
            if cluster_i != clusterAssment[i, 0]:
                clusterchanged = True
        print(centroids)
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 沿矩阵的列方向进行均值计算
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    # 返回所有的类质心与点分配结果
    return centroids, clusterAssment


def plt_my_KMeans():
    data = np.mat(loadDataSet(r'./10-K均值聚类算法/testSet.txt'))
    # 对数据进行聚类
    centroidsOfData, clusterAssmentOfData = my_KMeans(data, 4)
    # 数据集的数量
    m = np.shape(data)[0]
    # 画出数据的散点图
    plt.scatter(data[:, 0].A.reshape(m),
                data[:, 1].A.reshape(m),
                c=clusterAssmentOfData.A[:, 0].reshape(m))
    # 用红色的三角形符号画出聚类中心
    plt.scatter(centroidsOfData.A[:, 0],
                centroidsOfData.A[:, 1],
                c='red',
                marker='^')
    # 显示图片
    plt.show()


def biKmeans(dataSet, k, distMeas=distEclud):
    """
    [summary]:二分K-均值算法
        将所有点看成一个簇
        当簇数目小于k时
            对于每一个簇
                计算总误差
                在给定的簇上面进行K-均值聚类(k=2)
                计算将该簇一分为二之后的总误差
            选择使得误差最小的那个簇进行划分操作
                
    Arguments:
        dataSet {[mat]} -- 数据集
        k {[int]} -- 聚类数
    
    Keyword Arguments:
        distMeas  -- 距离算法 (default: {distEclud})
    
    Returns:
        centroids -- (k,n) 类质心
        clusterAssment -- (m,2) 点分配
    """
    m, n = np.shape(dataSet)
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 创建一个初始簇
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    cenList = [centroid0]  # 保留质心
    # 计算误差
    for j in range(m):
        clusterAssment[j, 1] = distMeas(np.mat(centroid0), dataSet[j, :])**2
    # 对簇进行划分
    while len(cenList) < k:
        lowestSSE = np.inf
        # 尝试划分每一簇
        for i in range(len(cenList)):
            #找出正在计算的簇
            ptscurrCluster = dataSet[np.nonzero(
                clusterAssment[:, 0].A == i)[0], :]
            # 对给定簇进行K-均值聚类
            centroidMat, splitClustAss = my_KMeans(ptscurrCluster, 2, distMeas)
            # 计算划分后的SSE(误差平方和)
            ssesplit = np.sum(splitClustAss[:, 1])
            # 计算剩余数据集的SSE(误差平方和)
            ssenotsplit = np.sum(
                clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])
            # print(ssesplit, ssenotsplit)

            if ssesplit + ssenotsplit < lowestSSE:
                bestCentToSplit = i
                bestnewCent = centroidMat
                # numpy中赋值都是将索引赋值，把数据真正赋值要用copy()
                bestClustAss = splitClustAss.copy()
                lowestSSE = ssenotsplit + ssesplit

        # 更新簇的分配结果
        bestClustAss[np.nonzero(
            bestClustAss[:, 0].A == 1)[0], 0] = len(cenList)
        bestClustAss[np.nonzero(
            bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        # print('the bestcenttosplit: ', bestCentToSplit)
        # print('len bestclustass: ', len(bestClustAss))
        # 要划分的簇的簇心坐标更新为其中一个簇心坐标
        cenList[bestCentToSplit] = bestnewCent[0, :].A.reshape(n)
        # 另一个簇心坐标要通过append添加进簇心坐标集合里
        cenList.append(bestnewCent[1, :].A.reshape(n))
        # reassign new clusters, and SSE
        clusterAssment[np.nonzero(
            clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss

    return np.mat(cenList), clusterAssment


def plt_biKmeans():
    data = np.mat(loadDataSet(r'./10-K均值聚类算法/testSet2.txt'))
    centroids, clusterAssment = biKmeans(data, 3)
    # 数据集的数量
    m = np.shape(data)[0]
    # 画出数据的散点图
    plt.scatter(data[:, 0].A.reshape(m),
                data[:, 1].A.reshape(m),
                c=clusterAssment.A[:, 0].reshape(m))
    # 用红色的三角形符号画出聚类中心
    plt.scatter(centroids.A[:, 0], centroids.A[:, 1], c='red', marker='+')
    # 显示图片
    plt.show()


def distSLC(vecA, vecB):
    # 使用球面余弦定理计算两点的距离
    a = np.sin(vecA[0, 1] * np.pi / 180) * np.sin(vecB[0, 1] * np.pi / 180)
    b = np.cos(vecA[0, 1] * np.pi / 180) * np.cos(
        vecB[0, 1] * np.pi / 180) * np.cos(np.pi *
                                           (vecB[0, 0] - vecA[0, 0]) / 180)
    return np.arccos(a + b) * 6371.0  # pi is imported with numpy


def clusterClubs(numClust=5):
    datList = []
    for line in open(r'./10-K均值聚类算法/places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = np.mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    # 基于图像创建矩阵
    imgP = plt.imread(r'./10-K均值聚类算法/Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0],
                    ptsInCurrCluster[:, 1].flatten().A[0],
                    marker=markerStyle,
                    s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0],
                myCentroids[:, 1].flatten().A[0],
                marker='+',
                s=300)
    plt.show()


if __name__ == '__main__':
    clusterClubs(5)
    # plt_my_KMeans()
    # plt_biKmeans()