'''
@version: 0.0.1
@Author: Huang
@dev: python3 vscode
@Date: 2019-11-06 20:00:17
@LastEditTime: 2019-11-07 19:33:18
@FilePath: \\机器学习实战\\08-线性回归\\regression.py
@Descripttion: 线性回归
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def loadDataSet(fileName):
    """
    [summary]:加载文件中的数据
    
    Arguments:
        fileName -- 文件名
    
    Returns:
        dataMat  -- 数据
        labelMat -- 标签
    """
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def plotDataSet():
    """
    [summary]:绘制数据集
    """
    dataMat, labelMat = loadDataSet(r'.\08-线性回归\ex0.txt')
    n = len(dataMat)  # 数据个数
    # 样本点
    xcord = []
    ycord = []
    for i in range(n):
        xcord.append(dataMat[i][1])
        ycord.append(labelMat[i])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord, ycord, s=20, c='blue', alpha=0.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()


def standRegres(xArr, yArr):
    """
    [summary]:计算最佳拟合直线
    
    Arguments:
        xArr  -- x数据集
        yArr  -- y标签集
    
    Returns:
        ws -- 回归系数
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    # 行列式不零是可逆矩阵
    if np.linalg.det(xTx) == 0.0:
        print("这个矩阵不可逆")
        return
    ws = xTx.I * (xMat.T * yMat)  # 求解公式
    return ws


def plotRegression():
    xArr, yArr = loadDataSet(r'.\08-线性回归\ex0.txt')  # 加载数据集
    ws = standRegres(xArr, yArr)  # 计算回归系数
    xMat = np.mat(xArr)  # 创建xMat矩阵
    yMat = np.mat(yArr)  # 创建yMat矩阵
    xCopy = xMat.copy()  # 深拷贝xMat矩阵
    xCopy.sort(0)  # 排序
    yHat = xCopy * ws  # 计算对应的y值
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 添加subplot
    ax.plot(xCopy[:, 1], yHat, c='red')  # 绘制回归曲线
    ax.scatter(xMat[:, 1].flatten().A[0],
               yMat.flatten().A[0],
               s=20,
               c='blue',
               alpha=.5)  # 绘制样本点
    plt.title('DataSet')  # 绘制title
    plt.xlabel('X')
    plt.show()


def lwlr(testPoint, xArr, yArr, k=1.0):
    """
    [summary]:局部加权线性回归,每次必须在整个数据集上运行
    
    Arguments:
        testPoint  -- 测试样本点
        xArr -- x数据集
        yArr -- y数据集
    
    Keyword Arguments:
        k {float} -- 高斯核的k (default: {1.0})
    
    Returns:
        ws -- 回归系数
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    # 创建权重对角矩阵
    weights = np.mat(np.eye((m)))
    for j in range(m):  # 权重值大小以指数级衰减
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k**2))
    xTx = xMat.T * (weights * xMat)

    if np.linalg.det(xTx) == 0.0:
        print("这个矩阵不可逆")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    """
    [summary]:为数据集中每个点调用lwlr()函数
    
    Arguments:
        testArr  -- 测试数据集
        xArr  --  x数据集
        yArr  --  y数据集
    
    Keyword Arguments:
        k {float} -- 高斯核的k (default: {1.0})
    
    Returns:
        ws -- 回归系数
    """
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):  # 对每个样本点进行预测
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def plotlwlrRegression():
    """
    [summary]:绘制多条局部加权回归曲线
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    xArr, yArr = loadDataSet(r'.\08-线性回归\ex0.txt')
    # 根据局部加权线性回归计算yHat
    yHat_1 = lwlrTest(xArr, xArr, yArr, 1.0)
    yHat_2 = lwlrTest(xArr, xArr, yArr, 0.01)
    yHat_3 = lwlrTest(xArr, xArr, yArr, 0.003)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    srtInd = xMat[:, 1].argsort(0)  # 排序，返回索引值
    xSort = xMat[srtInd][:, 0, :]
    fig, axs = plt.subplots(nrows=3,
                            ncols=1,
                            sharex=False,
                            sharey=False,
                            figsize=(10, 8))

    # 绘制回归曲线
    axs[0].plot(xSort[:, 1], yHat_1[srtInd], c='red')
    axs[1].plot(xSort[:, 1], yHat_2[srtInd], c='red')
    axs[2].plot(xSort[:, 1], yHat_3[srtInd], c='red')

    # 绘制样本点
    axs[0].scatter(xMat[:, 1].flatten().A[0],
                   yMat.flatten().A[0],
                   s=20,
                   c='blue',
                   alpha=.5)
    axs[1].scatter(xMat[:, 1].flatten().A[0],
                   yMat.flatten().A[0],
                   s=20,
                   c='blue',
                   alpha=.5)
    axs[2].scatter(xMat[:, 1].flatten().A[0],
                   yMat.flatten().A[0],
                   s=20,
                   c='blue',
                   alpha=.5)

    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0].set_title(u'局部加权回归曲线,k=1.0', FontProperties=font)
    axs1_title_text = axs[1].set_title(u'局部加权回归曲线,k=0.01', FontProperties=font)
    axs2_title_text = axs[2].set_title(u'局部加权回归曲线,k=0.003',
                                       FontProperties=font)
    plt.setp(axs0_title_text, size=8, weight='bold', color='red')
    plt.setp(axs1_title_text, size=8, weight='bold', color='red')
    plt.setp(axs2_title_text, size=8, weight='bold', color='red')
    plt.xlabel('X')
    plt.show()


def lwlrTestPlot(xArr, yArr, k=1.0):
    yHat = np.zeros(np.shape(yArr))  # easier for plotting
    xCopy = np.mat(xArr)
    xCopy.sort(0)
    for i in range(np.shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i], xArr, yArr, k)
    return yHat, xCopy


def rssError(yArr, yHatArr):
    """
    [summary]:计算平方和误差
    
    Arguments:
        yArr  -- 真实值
        yHatArr -- 预测值
    
    Returns:
        [type] -- [description]
    """
    return ((yArr - yHatArr)**2).sum()


def ridgeRegres(xMat, yMat, lam=0.2):
    """
    [summary]:岭回归 用来处理特征数多于样本数的情况,在估计中加入偏差
    
    Arguments:
        xMat  -- 数据矩阵
        yMat  -- 标签矩阵
    
    Keyword Arguments:
        lam {float} -- 缩减系数 (default: {0.2})
    
    Returns:
        ws -- 回归系数
    """
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("不可逆矩阵")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    """
    [summary]:岭回归测试,数据标准化
              所有特征减去各自的均值并除以方差
    
    Arguments:
        xArr  -- 数据数组
        yArr  -- 标签数组
    
    Returns:
        [type] -- [description]
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # 数据标准化
    yMean = np.mean(yMat, 0)  # 行与行操作，求均值
    yMat = yMat - yMean  # 数据减去均值

    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)  # 行与行操作，求方差
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30  # 30个不同的lambda测试
    # 初始回归系数矩阵
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):  # 改变lambda计算回归系数
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))  # lambda以e的指数变化
        wMat[i, :] = ws.T
    return wMat


def plotwMat():
    """
    [summary]:绘制岭回归系数矩阵
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    abX, abY = loadDataSet(r'.\08-线性回归\abalone.txt')
    redgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(redgeWeights)
    ax_title_text = ax.set_title(u'log(lambada)与回归系数的关系', FontProperties=font)
    ax_xlabel_text = ax.set_xlabel(u'log(lambada)', FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties=font)
    plt.setp(ax_title_text, size=20, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=10, weight='bold', color='black')
    plt.show()


def regularize(xMat, yMat):
    """
    [summary]:数据标准化
    
    Arguments:
        xMat  -- x数据集
        yMat  -- y数据集
    
    Returns:
        inxMat - 标准化后的x数据集
        inyMat - 标准化后的y数据集
    """
    inxMat = xMat.copy()  # 数据拷贝
    inyMat = yMat.copy()
    yMean = np.mean(inyMat, 0)  # 行与行操作，求均值
    inyMat = inyMat - yMean  # 数据减去均值
    inMeans = np.mean(inxMat, 0)  # 行与行操作，求均值
    inVar = np.var(inxMat, 0)  # 行与行操作，求方差
    inxMat = (inxMat - inMeans) / inVar  # 数据减去均值除以方差实现标准化
    return inxMat, inyMat


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    """
    [summary]:前向逐步线性回归
    
              数据标准化，使其分布满足0均值和单位方差
              在每轮迭代过程中:
              设置当前最小误差lowestError为正无穷
              对每个特征:
                增大或缩小:
                  改变一个系数得到一个新的W
                  计算新W下的误差
                  如果误差Error小于当前最小误差lowestError：设置Wbest等于当前的W
                将W设置为新的Wbest
    
    Arguments:
        xArr -- 数据数组
        yArr -- 标签数据
    Keyword Arguments:
        eps {float} -- 每次迭代需要调整的步长 (default: {0.01})
        numIt {int} -- 迭代次数 (default: {100})
    
    Returns:
        [type] -- [description]
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean  # can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))  # testing code remove
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


from bs4 import BeautifulSoup
from time import sleep
import json
import urllib.request


def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    """
    [summary]:从页面读取数据，生成retX和retY列表
    
    Arguments:
        retX {[type]} -- 数据X
        retY {[type]} -- 数据Y
        inFile {[type]} --  HTML文件
        yr {[type]} --  年份
        numPce {[type]} -- 乐高部件数目
        origPrc {[type]} -- 原价
    """

    # 打开并读取HTML文件
    with open(inFile, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html)
    i = 1
    # 根据HTML页面结构进行解析
    currentRow = soup.find_all('table', r="%d" % i)
    while (len(currentRow) != 0):
        currentRow = soup.find_all('table', r="%d" % i)
        title = currentRow[0].find_all('a')[1].text
        lwrTitle = title.lower()
        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].find_all('td')[3].find_all('span')
        if len(soldUnicde) == 0:
            print("商品 #%d 没有出售" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].find_all('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$', '')
            priceStr = priceStr.replace(',', '')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)
            # 去掉不完整的套装价格
            if sellingPrice > origPrc * 0.5:
                print("%d\t%d\t%d\t%f\t%f" %
                      (yr, numPce, newFlag, origPrc, sellingPrice))
                retX.append([yr, numPce, newFlag, origPrc])
                retY.append(sellingPrice)
        i += 1
        currentRow = soup.find_all('table', r="%d" % i)


def setDataCollect(retX, retY):
    # 2006年的乐高8288,部件数目800,原价49.99
    scrapePage(retX, retY, r'08-线性回归\setHtml\lego8288.html', 2006, 800, 49.99)
    scrapePage(retX, retY, r'08-线性回归\setHtml\lego10030.html', 2002, 3096,
               269.99)
    scrapePage(retX, retY, r'08-线性回归\setHtml\lego10179.html', 2007, 5195,
               499.99)
    scrapePage(retX, retY, r'08-线性回归\setHtml\lego10181.html', 2007, 3428,
               199.99)
    scrapePage(retX, retY, r'08-线性回归\setHtml\lego10189.html', 2008, 5922,
               299.99)
    scrapePage(retX, retY, r'08-线性回归\setHtml\lego10196.html', 2009, 3263,
               249.99)


def crossValidation(xArr, yArr, numVal=10):
    """
    [summary]:交叉验证岭回归
    
    Arguments:
        xArr {[type]} -- x数据集
        yArr {[type]} -- y数据集
    
    Keyword Arguments:
        numVal {int} -- 交叉验证次数 (default: {10})
    Returns:
        wMat -- 回归系数矩阵
    """
    m = len(yArr)  # 统计样本个数
    indexList = list(range(m))  # 生成索引值列表
    errorMat = np.zeros((numVal, 30))  # create error mat 30columns numVal rows
    for i in range(numVal):  # 交叉验证numVal次
        # 训练集
        trainX = []
        trainY = []
        # 测试集
        testX = []
        testY = []
        np.random.shuffle(indexList)  # 打乱次序
        for j in range(m):  # 划分数据集:90%训练集，10%测试集
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)  # 获得30个不同lambda下的岭回归系数
        for k in range(30):  # 遍历所有的岭回归系数
            matTestX = np.mat(testX)
            matTrainX = np.mat(trainX)  # 测试集
            meanTrain = np.mean(matTrainX, 0)  # 测试集均值
            varTrain = np.var(matTrainX, 0)  # 测试集方差
            matTestX = (matTestX - meanTrain) / varTrain  # 测试集标准化
            yEst = matTestX * np.mat(wMat[k, :]).T + np.mean(
                trainY)  # 根据ws预测y值
            errorMat[i, k] = rssError(yEst.T.A, np.array(testY))  # 统计误差
    meanErrors = np.mean(errorMat, 0)  # 计算每次交叉验证的平均误差
    minMean = float(min(meanErrors))  # 找到最小误差
    bestWeights = wMat[np.nonzero(meanErrors == minMean)]  # 找到最佳回归系数
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    meanX = np.mean(xMat, 0)
    varX = np.var(xMat, 0)
    unReg = bestWeights / varX  # 数据经过标准化，因此需要还原
    print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' %
          ((-1 * np.sum(np.multiply(meanX, unReg)) + np.mean(yMat)),
           unReg[0, 0], unReg[0, 1], unReg[0, 2], unReg[0, 3]))


def useStandRegres():
    """
    [summary]:使用简单的线性回归
    """
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    data_num, features_num = np.shape(lgX)
    lgX1 = np.mat(np.ones((data_num, features_num + 1)))
    lgX1[:, 1:5] = np.mat(lgX)
    ws = standRegres(lgX1, lgY)
    print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' %
          (ws[0], ws[1], ws[2], ws[3], ws[4]))


def usesklearn():
    """
    [summary]:使用sklearn
    """
    from sklearn import linear_model
    reg = linear_model.Ridge(alpha=.5)
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    reg.fit(lgX, lgY)
    print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' %
          (reg.intercept_, reg.coef_[0], reg.coef_[1], reg.coef_[2],
           reg.coef_[3]))


if __name__ == '__main__':
    usesklearn()