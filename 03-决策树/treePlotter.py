import matplotlib.pyplot as plt

# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def createPlot():
    fig = plt.figure(10, facecolor="white")
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('a decison node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


# 绘制带箭头的注解
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(
        nodeTxt,
        xy=parentPt,  # 设置箭头尖的坐标
        xycoords="axes fraction",
        xytext=centerPt,  # 设置注释内容显示的起始位置
        textcoords="axes fraction",
        va="center",  # 垂直方向
        ha="center",  # 水平方向
        bbox=nodeType,  # 对方框的设置
        arrowprops=arrow_args)
