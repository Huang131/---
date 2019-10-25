"""
@Descripttion: kNN: k Nearest Neighbors
 优点:分类数据最简单最有效的算法
 缺点:无法给出任何数据的基础结构信息,无法知晓平均实例样本和典型实例样本具有的特诊
@version: 0.0.1
@Author: Huang
@dev: python3 vscode
@Date: 2019-06-01 20:28:37
@LastEditors: Huang
@LastEditTime: 2019-10-26 01:47:09
"""
import numpy as np
import operator
from os import listdir
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as kNN


def createDataSet():
    """
    创建数据集
    
    Returns:
        group - 数据集
        labels - 分类标签
    """
    # 四组二维特征
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 81]])
    # 四组特诊的标签
    labels = ["爱情片", "爱情片", "动作片", "动作片"]
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    [summary]:KNN算法,分类器
    
    Arguments:
        inX {[type]} -- 用于分类的数据(测试集)
        dataSet {[type]} -- 用于训练的数据(训练集)
        labels {[type]} -- 分类标签
        k {[type]} -- 选择距离最小的k个点
    
    Returns:
        sortedClassCount[0][0] -- 分类结果
    """

    # 距离计算
    dataSetSize = dataSet.shape[0]  # 训练集行数
    # *行向量方向上重复inX共dataSetSize次,列向量方向上重复inX共1次
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    # *sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5  # 欧式距离
    sortedDistIndicies = distances.argsort()  # 返回数组值从小到大的索引值
    # 定一个记录类别次数的字典
    classCount = {}
    # 选择距离最小的K个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # key=operator.itemgetter(0)根据字典的键进行排序
    # key=operator.itemgetter(1)根据字典的值进行排序
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True
    )
    return sortedClassCount[0][0]


def file2matrix(filename):
    """
    [summary]:打开并解析文件，对数据进行分类：1代表不喜欢,2代表魅力一般,3代表极具魅力
    
    Arguments:
        filename {[type]} -- 文件名
    
    Returns:
        returnMat - 特征矩阵
        classLabelVector - 分类Label向量
    """

    fr = open(filename)
    arrayLines = fr.readlines()  # 读取文件所有内容
    numberOfLines = len(arrayLines)  # 得到文件行数
    returnMat = np.zeros((numberOfLines, 3))  # 创建返回NumPy矩阵
    classLabelVector = []  # 返回的分类标签向量
    index = 0
    # 解析文件数据到文本
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split("\t")
        returnMat[index, :] = listFromLine[0:3]

        # 根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listFromLine[-1] == "didntLike":
            classLabelVector.append(1)
        elif listFromLine[-1] == "smallDoses":
            classLabelVector.append(2)
        elif listFromLine[-1] == "largeDoses":
            classLabelVector.append(3)

        index += 1

    return returnMat, classLabelVector


def showdatas(datingDataMat, datingLabels):
    """
    [summary]:可视化数据
    
    Args:
        datingDataMat ([type]): 特征矩阵
        datingLabels ([type]): 分类Label
    """
    # 设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(
        nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8)
    )

    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append("black")
        if i == 2:
            LabelsColors.append("orange")
        if i == 3:
            LabelsColors.append("red")
    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(
        x=datingDataMat[:, 0],
        y=datingDataMat[:, 1],
        color=LabelsColors,
        s=15,
        alpha=0.5,
    )
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(
        u"每年获得的飞行常客里程数与玩视频游戏所消耗时间占比", FontProperties=font
    )
    axs0_xlabel_text = axs[0][0].set_xlabel(u"每年获得的飞行常客里程数", FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u"玩视频游戏所消耗时间占", FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight="bold", color="red")
    plt.setp(axs0_xlabel_text, size=7, weight="bold", color="black")
    plt.setp(axs0_ylabel_text, size=7, weight="bold", color="black")

    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(
        x=datingDataMat[:, 0],
        y=datingDataMat[:, 2],
        color=LabelsColors,
        s=15,
        alpha=0.5,
    )
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(
        u"每年获得的飞行常客里程数与每周消费的冰激淋公升数", FontProperties=font
    )
    axs1_xlabel_text = axs[0][1].set_xlabel(u"每年获得的飞行常客里程数", FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u"每周消费的冰激淋公升数", FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight="bold", color="red")
    plt.setp(axs1_xlabel_text, size=7, weight="bold", color="black")
    plt.setp(axs1_ylabel_text, size=7, weight="bold", color="black")

    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(
        x=datingDataMat[:, 1],
        y=datingDataMat[:, 2],
        color=LabelsColors,
        s=15,
        alpha=0.5,
    )
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(
        u"玩视频游戏所消耗时间占比与每周消费的冰激淋公升数", FontProperties=font
    )
    axs2_xlabel_text = axs[1][0].set_xlabel(u"玩视频游戏所消耗时间占比", FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u"每周消费的冰激淋公升数", FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight="bold", color="red")
    plt.setp(axs2_xlabel_text, size=7, weight="bold", color="black")
    plt.setp(axs2_ylabel_text, size=7, weight="bold", color="black")
    # 设置图例
    didntLike = mlines.Line2D(
        [], [], color="black", marker=".", markersize=6, label="didntLike"
    )
    smallDoses = mlines.Line2D(
        [], [], color="orange", marker=".", markersize=6, label="smallDoses"
    )
    largeDoses = mlines.Line2D(
        [], [], color="red", marker=".", markersize=6, label="largeDoses"
    )
    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # 显示图片
    plt.show()


def autoNorm(dataSet):
    """
    [summary]:对数据进行归一化
    
    Arguments:
        dataSet {[type]} -- 特征矩阵
    
    Returns:
            normDataSet - 归一化后的特征矩阵
            ranges - 数据范围
            minVals - 数据最小值
    """
    minVals = dataSet.min(0)  # 每列最小值
    maxVals = dataSet.max(0)  # 每列最大值
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))  # 特征值相除
    return normDataSet, ranges, minVals


def datingClassTest():
    """
    [summary]:分类器测试函数
    
    Returns:
        normDataSet - 归一化后的特征矩阵
        ranges - 数据范围
        minVals - 数据最小值
    """

    # 获取所有数据的百分之十
    hoRatio = 0.10
    filename = r"./02-K-近邻算法/datingTestSet.txt"
    # 将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
    datingDataMat, datingLabels = file2matrix(filename)
    # 数据归一化,返回归一化后的矩阵,数据范围,数据最小值
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]  # 行数
    # 百分之十的测试数据的个数
    numTestVecs = int(m * hoRatio)
    # 分类错误计数
    errorCount = 0.0

    for i in range(numTestVecs):
        # 前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        classifierResult = classify0(
            normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3
        )  # 向量 训练集 标签 k
        print("分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("错误率:%f%%" % (errorCount / float(numTestVecs)))


def classifyPerson():
    """
    [summary]:预测对某人的喜欢程度
    """
    resultList = ["讨厌", "有些喜欢", "非常喜欢"]
    # 三维特征用户输入
    percentTats = float(input("玩视频游戏所耗时间百分比:"))
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    filename = r"./02-K-近邻算法/datingTestSet.txt"
    # 打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    # 训练集归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 生成NumPy数组,测试集
    inArr = np.array([ffMiles, percentTats, iceCream])
    # 返回分类结果
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("你可能%s这个人" % (resultList[classifierResult - 1]))


def img2vector(filename):
    """
    将32 * 32图像转换为1 * 1024向量
    """
    # 创建1x1024零向量
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        # 每一行的前32个元素依次添加到returnVect中
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    """
    手写数字识别系统的测试代码
    """
    # 测试集的Labels
    hwLabels = []
    # 返回trainingDigits目录下的文件名
    trainingFileList = listdir("./02-K-近邻算法/digits/trainingDigits")
    # 返回文件夹下文件的个数
    m = len(trainingFileList)
    # 初始化训练的Mat矩阵,测试集
    trainingMat = np.zeros((m, 1024))
    # 从文件名中解析出训练集的类别
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split(".")[0]
        # 获得分类的数字
        classNumStr = int(fileStr.split("_")[0])
        hwLabels.append(classNumStr)
        # 将每一个文件的1x1024数据存储到trainingMat矩阵中
        trainingMat[i, :] = img2vector(
            "./02-K-近邻算法/digits/trainingDigits/%s" % fileNameStr
        )
    # !构建kNN分类器
    neigh = kNN(n_neighbors=3, algorithm="auto")
    # !拟合模型, trainingMat为测试矩阵,hwLabels为对应的标签
    neigh.fit(trainingMat, hwLabels)
    # 返回testDigits目录下的文件列表
    testFileList = listdir("./02-K-近邻算法/digits/testDigits")
    # 错误检测计数
    errorCount = 0.0
    # 测试数据的数量
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        vectorUnderTest = img2vector("./02-K-近邻算法/digits/testDigits/%s" % fileNameStr)
        # classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        classifierResult = neigh.predict(vectorUnderTest)
        print("分类结果为:%d,实际结果为:%d" % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print("错误次数: %d" % errorCount)
    print("错误率: %f%%" % (errorCount / float(mTest) * 100))


if __name__ == "__main__":
    handwritingClassTest()

