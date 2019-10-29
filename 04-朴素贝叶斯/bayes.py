"""
@Descripttion: 基于贝叶斯决策理论的分类方法
优点：在数据较少的情况下仍然有效，可以处理多类别问题
缺点：对于输入数据的准备方式较为敏感
适用数据类型：标称型数据
@version: 0.0.1
@Author: Huang
@dev: python3 vscode
@Date: 2019-10-27 00:58:34
@LastEditors: Huang
@LastEditTime: 2019-10-27 01:09:54
"""

import numpy as np
import re


def loadDataSet():
    """
    [summary]:创建实验样本
    
    Returns:
        postingList - 实验样本切分的词条
        classVec - 类别标签向量
    """

    postingList = [
        ["my", "dog", "has", "flea", "problems", "help", "please"],  # 切分的词条
        ["maybe", "not", "take", "him", "to", "dog", "park", "stupid"],
        ["my", "dalmation", "is", "so", "cute", "I", "love", "him"],
        ["stop", "posting", "stupid", "worthless", "garbage"],
        ["mr", "licks", "ate", "my", "steak", "how", "to", "stop", "him"],
        ["quit", "buying", "worthless", "dog", "food", "stupid"],
    ]
    classVec = [0, 1, 0, 1, 0, 1]  # 类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList, classVec


def createVocabList(dataSet):
    """
    [summary]:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
    
    Arguments:
        dataSet -- 整理的样本数据集
    Returns:
        vocabSet - 返回不重复的词条列表，也就是词汇表
    """
    vocabSet = set([])  # 创建一个空的列表集合
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 取并集
    return list(vocabSet)


# 词集模型:将词语是否出现作为一个特征
def setOfWords2Vec(vocabList, inputSet):
    """
    [summary]:将inputSet向量化,通过词汇表判断元素是否存在,向量的每个元素为1或0
    
    Arguments:
        vocabList {[type]} -- createVocabList返回的列表
        inputSet {[type]} -- 切分的词条列表
    
    Returns:
        returnVec -- 文档向量,词集模型
    """
    returnVec = [0] * len(vocabList)  # 长度为词汇表长度
    for word in inputSet:  # 遍历每个词条
        returnVec[vocabList.index(word)] = 1
    else:
        print("the word: %s is not in my Vocabulary!" %
              word)  # for 循环正常结束,执行else中的语句
    return returnVec  # 返回文档向量


# 词袋模型:每个单词可以出现多次
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """
    [summary]:朴素贝叶斯分类器训练函数
    
    Arguments:
    trainMatrix -- 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
    trainCategory -- 训练类别标签向量，即loadDataSet返回的classVec
    
    Returns:
            p0Vect - 侮辱类的条件概率数组
            p1Vect - 非侮辱类的条件概率数组
            pAbusive - 文档属于侮辱类的概率(1:侮辱性词汇)
    """
    numTrainDocs = len(trainMatrix)  # 计算训练的文档数目
    numWords = len(trainMatrix[0])  # 计算每篇文档的词条数
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 文档属于侮辱类的概率
    p0Num = np.ones(
        numWords)  # 创建numpy.ones数组,词条出现数初始化为1，防止其中一个概率为零后，导致相乘后的总体概率为零
    p1Num = np.ones(numWords)
    p0Denom = 2.0  # 分母初始化为2.0,拉普拉斯平滑
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[
                i] == 1:  # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:  # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p0Vect = np.log(p0Num / p0Denom)  # 对每个元素做除法,存储每个单词属于非侮辱类的概率
    p1Vect = np.log(p1Num / p1Denom)

    return p0Vect, p1Vect, pAbusive  # 返回属于非侮辱类的条件概率数组，属于侮辱类的条件概率数组，文档属于侮辱类的概率


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    [summary]:分类函数。这里很巧妙的利用了numpy数组,vec2Classify中对应出现的词汇为1，其他为零。贝叶斯公式在实际运用中,因为分母都一样,可以忽略计算。因为计算机对小数的乘法容易出现溢出错误,所以转化为log相加的形式
    
    Arguments:
        vec2Classify - 待分类的词条向量数组
        p0Vec - 侮辱类的条件概率数组
        p1Vec -非侮辱类的条件概率数组
        pClass1 - 文档属于侮辱类的概率
    
    Returns:
        0 - 属于非侮辱类
        1 - 属于侮辱类
    """
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)

    return 1 if p1 > p0 else 0


def testingNB():
    """
    [summary]:测试函数
    """
    listOPosts, listClasses = loadDataSet()  # 创建实验样本
    myVocabList = createVocabList(listOPosts)  # 创建词汇表
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))  # 将实验样本向量化
    p0V, p1V, pAb = trainNB0(np.array(trainMat),
                             np.array(listClasses))  # 训练朴素贝叶斯分类器
    print("p0v", p0V)
    print("p1v", p1V)
    print("pAb", pAb)
    testEntry = ["love", "my", "dalmation"]  # 测试样本1
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))  # 测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, "属于侮辱类")  # 执行分类并打印分类结果
    else:
        print(testEntry, "属于非侮辱类")  # 执行分类并打印分类结果
    testEntry = ["stupid", "garbage"]  # 测试样本2

    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))  # 测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, "属于侮辱类")  # 执行分类并打印分类结果
    else:
        print(testEntry, "属于非侮辱类")  # 执行分类并打印分类结果


def textParse(bigString):
    """
    [summary]:将字符串解析为长度大于2的小写字符串列表
    
    Arguments:
        bigString {[type]} -- [description]
    
    Returns:
        [] -- 字符串列表
    """
    listOfTokens = re.split(r"\W+", bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    """
    [summary]:对垃圾邮件进行自动化处理
    """
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):  # 遍历txt文件
        # 读取每个垃圾邮件，并字符串转换成字符串列表
        wordList = textParse(open("./04-朴素贝叶斯/email/spam/%d.txt" % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # 标记垃圾邮件，1表示垃圾文件
        # 读取每个非垃圾邮件，并字符串转换成字符串列表
        wordList = textParse(open("./04-朴素贝叶斯/email/ham/%d.txt" % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)  # 标记非垃圾邮件，0表示垃圾文件

    vocabList = createVocabList(docList)  # 创建词汇表，不重复
    trainingSet = list(range(50))  # 创建存储训练集的索引值的列表和测试集的索引值的列表
    testSet = []
    for i in range(10):  # 从50个邮件中，随机挑选出40个作为训练集,10个做测试集
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]  # 在训练集列表中删除添加到测试集的索引值

    # 创建训练集矩阵和训练集类别标签系向量
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:  # 遍历训练集
        trainMat.append(setOfWords2Vec(vocabList,
                                       docList[docIndex]))  # 将生成的词集模型添加到训练矩阵中
        trainClasses.append(classList[docIndex])  # 将类别添加到训练集类别标签系向量中
    p0V, p1V, pSpam = trainNB0(np.array(trainMat),
                               np.array(trainClasses))  # 训练朴素贝叶斯模型
    errorCount = 0  # 错误分类计数

    for docIndex in testSet:  # 遍历测试集
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])  # 测试集的词集模型
        if classifyNB(np.array(wordVector), p0V, p1V,
                      pSpam) != classList[docIndex]:
            errorCount += 1
            print("分类错误的测试集：", docList[docIndex])
    print("错误率:%.2f%%" % (float(errorCount) / len(testSet) * 100))


if __name__ == "__main__":
    spamTest()
