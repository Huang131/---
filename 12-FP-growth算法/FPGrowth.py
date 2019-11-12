'''
@version: 0.0.1
@Author: Huang
@dev: python3 vscode
@Date: 2019-11-12 09:29:53
@LastEditTime: 2019-11-12 12:24:30
@FilePath: \\机器学习实战\\12-FP-growth算法\\FPGrowth.py
@Descripttion: 只需对数据库进行两次扫描,第一次对所有元素项出现次数进行统计;第二次只考虑频繁元素,它能够更为高效的挖掘数据
'''


class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        # 节点元素名称
        self.name = nameValue
        # 出现次数
        self.count = numOccur
        # 指向下一个相似节点的指针
        self.nodeLink = None
        # 指向父节点的指针
        self.parent = parentNode
        # 指向子节点的字典，以子节点的元素名称为键，指向子节点的指针为值
        self.children = {}

    # 增加节点的出现次数值
    def inc(self, numOccur):
        self.count += numOccur

    # 输出节点和子节点的FP树结构
    def disp(self, ind=1):
        print(' ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)


def createTree(dataSet, minSup=1):
    """
    [summary]:将数据集转化为FP树
    1. 遍历数据集，统计各元素项出现次数，创建头指针表
    2. 移除头指针表中不满足最小值尺度的元素项
    3. 第二次遍历数据集，创建FP树。对每个数据集中的项集：
        3.1 初始化空FP树
        3.2 对每个项集进行过滤和重排序
        3.3 使用这个项集更新FP树，从FP树的根节点开始：
            3.3.1 如果当前项集的第一个元素项存在于FP树当前节点的子节点中，则更新这个子节点的计数值
            3.3.2 否则，创建新的子节点，更新头指针表
            3.3.3 对当前项集的其余元素项和当前元素项的对应子节点递归3.3的过程
          
    Arguments:
        dataSet  -- 数据集
    
    Keyword Arguments:
        minSup {int} -- 最小支持度 (default: {1})
    
    Returns:
        [type] -- [description]
    """
    # 第一次遍历数据集，创建头指针表
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 移除不满足最小支持度的元素项
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del (headerTable[k])
    # 空元素集，返回空
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None
    # 增加一个数据项，用于存放指向相似元素项指针
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    # 创造根节点
    retTree = treeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
        # 对一个项集tranSet，记录其中每个元素项的全局频率，用于排序
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            # 根据全局频率对每个事务中的元素进行排序
            orderedItems = [
                v[0] for v in sorted(
                    localD.items(), key=lambda p: p[1], reverse=True)
            ]
            # 使用排序后的元素项集对树进行填充
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable


def updateTree(items, inTree, headerTabel, count):
    # 有该元素项时计数值+1
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        # 没有这个元素项时创建一个新节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTabel[items[0]][1] == None:
            headerTabel[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTabel[items[0]][1], inTree.children[items[0]])
    # 对剩下的元素项迭代调用updateTree函数
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTabel, count)


def updateHeader(nodeToTest, targetNode):
    # 获取头指针表中该元素项对应的单链表的尾节点,然后将其指向新节点targetNode
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def loadSimpDat():
    # 加载数据集
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'], ['z'],
               ['r', 'x', 'n', 'o', 's'], ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):
    # 生成数据集
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


def test_fp():
    simpDat = loadSimpDat()
    initSet = createInitSet(simpDat)
    myFPtree, myHeaderTab = createTree(initSet, 3)
    myFPtree.disp()


def ascendTree(leafNode, prefixPath):
    # 迭代上溯整课树
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):  # treeNode comes from header table
    # 创建前缀路径
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


def test_pre():
    simpDat = loadSimpDat()
    initSet = createInitSet(simpDat)
    myFPtree, myHeaderTab = createTree(initSet, 3)
    condPats = findPrefixPath('r', myHeaderTab['r'][1])
    print(condPats)


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    """
    [summary]:递归查找频繁项集
    
    Arguments:
        inTree {[type]} -- [description]
        headerTable {[type]} -- [description]
        minSup {[type]} -- [description]
        preFix {[type]} -- [description]
        freqItemList  -- 频繁项集列表
    """
    # 对头指针表中的元素项按其出现频率进行排序(默认从小到大)
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]
    for basePat in bigL:  # start from bottom of header table
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        myCondTree, myHead = createTree(condPattBases, minSup)
        if myHead != None:
            print('conditional tree for: ', newFreqSet)
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)


def test_mineTree():
    simpDat = loadSimpDat()
    initSet = createInitSet(simpDat)
    myFPtree, myHeaderTab = createTree(initSet, 3)
    freqItems = []
    mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)


if __name__ == '__main__':
    paresdDat = [
        line.split()
        for line in open(r'.\12-FP-growth算法\kosarak.dat').readlines()
    ]

    initSet = createInitSet(paresdDat)
    myFPtree, myHeaderTab = createTree(initSet, 100000)
    myFreaList = []
    mineTree(myFPtree, myHeaderTab, 100000, set([]), myFreaList)
    print('len:', len(myFreaList), 'myFreaList', myFreaList)