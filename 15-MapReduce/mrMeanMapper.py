'''
@version: 0.0.1
@Author: Huang
@dev: python3 vscode
@Date: 2019-11-12 22:53:08
@LastEditTime: 2019-11-12 23:08:24
@FilePath: \\机器学习实战\\15-MapReduce\\mrMeanMapper.py
@Descripttion: 分布式计算均值和方差的mapper
'''

import sys
import numpy as np


# 读取数据
def read_input(file):
    for line in file:
        yield line.rstrip()


input = read_input(sys.stdin)  # creates a list of input lines
input = [float(line) for line in input]  # overwrite with floats
numInputs = len(input)
input = np.mat(input)
sqInput = np.power(input, 2)

# output size, mean, mean(square values)
print("%d\t%f\t%f" % (numInputs, np.mean(input), np.mean(sqInput)))
# print(sys.stderr, "report: still alive")
