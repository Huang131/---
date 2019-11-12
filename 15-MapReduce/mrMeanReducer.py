'''
@version: 0.0.1
@Author: Huang
@dev: python3 vscode
@Date: 2019-11-12 23:02:44
@LastEditTime: 2019-11-12 23:25:23
@FilePath: \\机器学习实战\\15-MapReduce\\mrMeanReducer.py
@Descripttion: 分布式计算均值和方差的reducer
'''

import sys


def read_input(file):
    for line in file:
        yield line.rstrip()


input = read_input(sys.stdin)  # creates a list of input lines

# split input lines into separate items and store in list of lists
mapperOut = [line.split('\t') for line in input]

# 均值
cumVal = 0.0
# 平方和均值
cumSumSq = 0.0
# 大小
cumN = 0.0
for instance in mapperOut:
    nj = float(instance[0])
    cumN += nj
    cumVal += nj * float(instance[1])
    cumSumSq += nj * float(instance[2])

# calculate means
mean = cumVal / cumN
meanSq = cumSumSq / cumN

# output size, mean, mean(square values)
print("%d\t%f\t%f" % (cumN, mean, meanSq))
print(sys.stderr, "report: still alive")