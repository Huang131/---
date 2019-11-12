'''
@version: 0.0.1
@Author: Huang
@dev: python3 vscode
@Date: 2019-11-12 12:40:12
@LastEditTime: 2019-11-12 21:04:40
@FilePath: \\机器学习实战\\13-PCA\\PCA.py
@Descripttion: 在低维下,数据更容易进行处理,相关特征可能在数据中更明确地显示出来。PCA降维是把数据从原来的坐标系转换到了新的坐标系,新坐标系的选择是由数据本身决定的。
'''
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName=r'13-PCA\testSet.txt'):
    with open(fileName) as fr:
        stringArr = [line.strip().split() for line in fr.readlines()]
        datArr = [list(map(float, line)) for line in stringArr]
    return np.mat(datArr)


def pca(dataMat, topNfeat=9999999):
    """
    [summary]:
        去除平均值
        计算协方差矩阵
        计算协方差矩阵的特征值和特征向量
        将特征值从大到小排序
        保留最上面的N个特征向量
        将数据转换到上述N个特征向量构建的新空间中
    Arguments:
        dataMat  -- 数据集
    
    Keyword Arguments:
        topNfeat {int} -- 选取的特征数 (default: {9999999})
    
    Returns:
        [type] -- [description]
    """
    meanVals = np.mean(dataMat, axis=0)
    # 去除平均值
    meanRemoved = dataMat - meanVals
    # 计算协方差矩阵及其特征值
    covMat = np.cov(meanRemoved, rowvar=0)  # 0-column nonzero-row
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    # 对特征值进行排序,argsort返回索引
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[-1:-(topNfeat + 1):-1]
    redEigVects = eigVects[:, eigValInd]
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


def test_pca():
    dataMat = loadDataSet(r'13-PCA\testSet.txt')
    lowDMat, reconMat = pca(dataMat, 1)
    print('lowDMat shape:', lowDMat.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0],
               dataMat[:, 1].flatten().A[0],
               marker='^',
               s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0],
               reconMat[:, 1].flatten().A[0],
               marker='o',
               s=50,
               c='red')
    plt.show()


def replaceNanWithMean():
    datMat = loadDataSet(r'13-PCA\secom.data')
    numFeat = np.shape(datMat)[1]
    for i in range(numFeat):
        # 计算所有非NaN的平均值 np.isnan返回结果数组,元素为np.nan的位置对应为True
        meanVal = np.mean(datMat[np.nonzero(~np.isnan(datMat[:, i].A))[0], i])
        datMat[np.nonzero(np.isnan(datMat[:, i]))[0], i] = meanVal
    return datMat


if __name__ == '__main__':
    # test_pca()
    dataMat = replaceNanWithMean()
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    # 计算协方差矩阵 rowvar=False表示每列作为观测量
    covMat = np.cov(meanRemoved, rowvar=False)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    print(eigVals)
