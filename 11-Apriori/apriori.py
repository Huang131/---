'''
@version: 0.0.1
@Author: Huang
@dev: python3 vscode
@Date: 2019-11-11 11:24:22
@LastEditTime: 2019-11-11 16:56:01
@FilePath: \\机器学习实战\\11-Apriori\\apriori.py
@Descripttion: 如果一个元素项是不频繁的，那么那些包含该元素的超集也是不频繁的。Apriori算法从单元素项集开始，通过组合满足最小支持度要求的项集来形成更大的集合。支持度用来度量一个集合在原始数据中出现的频率
'''


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    """
    [summary]:构建集合C1,C1是大小为1的所有候选项集的集合
    
    Arguments:
        dataSet List[List] -- 数据集
    
    Returns:
        List[frozenset]
    """
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if [item] not in C1:
                C1.append([item])  # 注意，添加的是该项的列表,为每个物品项构建一个集合
    C1.sort()
    # 对C1中的每个项构建一个不变集合,frozenset可以作为字典键值使用,set不能
    return list(map(frozenset, C1))


def scanD(Data, Ck, minSupport):
    """
    [summary]:满足最低要求的项集生成集合L1
    
    Arguments:
        Data List[set] -- 数据集
        Ck List[frozenset] -- 候选项集列表Ck
        minSupport float -- 最小支持度
    
    Returns:
        retList -- 满足最低要求的项集
        supportData -- 包含支持度的字典
    """
    ssCnt = {}
    for tid in Data:
        for can in Ck:  # 遍历候选项
            if can.issubset(tid):  # 判断集合can中每一项元素是否都在tid中
                ssCnt[can] = ssCnt.get(can, 0) + 1
    numItems = float(len(Data))
    retList = []
    supportData = {}
    # 计算所有项集的支持度
    for key in ssCnt.keys():
        support = ssCnt[key] / numItems
        supportData[key] = support
        if support >= minSupport:
            retList.insert(0, key)
    return retList, supportData


def test_scanD():
    dataSet = loadDataSet()
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, suppData0 = scanD(D, C1, 0.5)
    print("L1:", L1)
    print("suppData0:", suppData0)


def aprioriGen(Lk, k):
    """
    [summary]:组合,向上合并。根据Lk和k输出所有可能的候选集Ck

    Arguments:
        Lk List[set] -- 频繁项集列表
        k int -- 返回的项集元素个数
    
    Returns:
        retList -- 元素两两合并的数据集
    """
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            # 在限制项数为k的情况下，只有前k-2项相同，才能生成新的候选项
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])  # 取并集
    return retList


def apriori(dataSet, minSupport=0.5):
    """
    [summary]
    当集合中项的个数大于0时
        构建一个k个项组成的候选项集的列表
        检查数据以确认每个项集都是频繁的
        保留频繁项集并构建k+1项组成的候选项集的列表
        
    Arguments:
        dataSet  -- 数据集
    
    Keyword Arguments:
        minSupport {float} -- 最小支持度 (default: {0.5})
    
    Returns:
        L  -- 频繁项集的全集
        supportData -- 所有元素和支持度的全集
    """
    # 对原始数据集进行去重,排序。将元素转换为frozenset
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    # 计算候选数据集C1在D中的支持度,返回支持度大于minSupport的数据
    L1, supportData = scanD(D, C1, minSupport)
    # L 会包含 L1、L2...
    L = [L1]
    k = 2
    while (len(L[k - 2]) > 0):  # 迭代创建L1,L2,...直到集合为空为止
        Ck = aprioriGen(L[k - 2], k)
        # 扫描数据集,从Ck得到Lk
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


def test_apriori():
    dataSet = loadDataSet()
    L, suppData = apriori(dataSet)
    print("L:", L)
    print("suppData:", suppData)


def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    """
    [summary]:计算可信度,支持度定义: a -> b = support(a | b) / support(a).
    
    Arguments:
        freqSet  -- 频繁项集中的元素，例如: frozenset([2, 3, 5])
        H  -- 繁项集中的元素的集合 例如: [frozenset([2]), frozenset([3]), frozenset([5])]
        supportData {dict} -- 支持度字典
        brl {[type]} -- 关联规则列表的空数组
    
    Keyword Arguments:
        minConf {float} -- 最小可信度 (default: {0.7})
    
    Returns:
        List -- 满足最小可信度要求的规则列表
    """
    prunedH = []
    for conseq in H:
        # 假设freqSet=frozenset([1, 3]), conseq=[frozenset([1])],那么frozenset([1])至frozenset([3])的可信度为
        # supportData[freqSet]/supportData[freqSet-conseq]
        # = supportData[frozenset([1, 3])] / supportData[frozenset([1])]
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            # 添加到规则里，注意py里list是引用传递。brl 是前面通过检查的 bigRuleList
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    """
    [summary]:递归计算频繁项集的规则
    
    Arguments:
        freqSet  -- 频繁项集中的元素，例如: frozenset([2, 3, 5])
        H {[type]} -- 出现在规则右部的元素列表 例如: [frozenset([2]), frozenset([3]), frozenset([5])]
        supportData -- 支持度字典
        brl {[type]} -- 关联规则列表的数组
    
    Keyword Arguments:
        minConf {float} -- 最小可信度 (default: {0.7})
    """
    # H[0]是freqSet 的元素组合的第一个元素，并且H中所有元素的长度都一样，长度由aprioriGen(H, m+1) 这里的 m + 1 来控制
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        # 生成 m+1 个长度的所有可能的 H 中的组合
        Hmp1 = aprioriGen(H, m + 1)
        # 返回可信度大于minConf的集合
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):  # 满足最小可信度要求的规则列表多于1,则递归来判断是否可以进一步组合这些规则
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


def generateRules(L, supportData, minConf=0.7):
    """
    [summary]:生成关联规则
    
    Arguments:
        L {[type]} -- 频繁项集列表
        supportData {[type]} -- 频繁项集支持度的字典
    
    Keyword Arguments:
        minConf {float} -- 最小置信度 (default: {0.7})
    
    Returns:
        [List] -- 信度规则列表（关于 (A->B+置信度) 3个字段的组合）
    """
    bigRuleList = []
    for i in range(1, len(L)):
        # 两个及以上的才能有关联
        for freqSet in L[i]:
            # 假设：freqSet= frozenset([1, 3]), H1=[frozenset([1]), frozenset([3])
            H1 = [frozenset([item]) for item in freqSet]
            # 如果频繁项集元素数目超过2,那么会考虑对它做进一步的合并
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:  # 第一层时，后件数为1
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def test_rules():
    dataSet = loadDataSet()
    L, suppData = apriori(dataSet, minSupport=0.5)
    print("L:", L)
    print("suppData:", suppData)
    rules = generateRules(L, suppData, minConf=0.7)
    print("rules:", rules)


def test_mushroom():
    mushDatSet = [
        line.split()
        for line in open(r'./11-Apriori/mushroom.dat').readlines()
    ]

    L, suppData = apriori(mushDatSet, minSupport=0.3)
    for item in L[3]:
        if item.intersection('2'):
            print(item)


if __name__ == '__main__':
    test_mushroom()
