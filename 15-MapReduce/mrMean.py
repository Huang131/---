'''
@version: 0.0.1
@Author: Huang
@dev: python3 vscode
@Date: 2019-11-12 23:23:56
@LastEditTime: 2019-11-12 23:27:58
@FilePath: \\机器学习实战\\15-MapReduce\\proximalSVM.py
'''
from mrjob.job import MRJob


class MRmean(MRJob):
    def __init__(self, *args, **kwargs):
        super(MRmean, self).__init__(*args, **kwargs)
        self.inCount = 0
        self.inSum = 0
        self.inSqSum = 0

    def map(self, key, val):  # 接收输入数据流
        if False:
            yield
        inVal = float(val)
        self.inCount += 1
        self.inSum += inVal
        self.inSqSum += inVal * inVal

    def map_final(self):  # 所有输入到达后开始处理
        mn = self.inSum / self.inCount
        mnSq = self.inSqSum / self.inCount
        yield (1, [self.inCount, mn, mnSq])

    def reduce(self, key, packedValues):
        cumVal = 0.0
        cumSumSq = 0.0
        cumN = 0.0
        for valArr in packedValues:  # get values from streamed inputs
            nj = float(valArr[0])
            cumN += nj
            cumVal += nj * float(valArr[1])
            cumSumSq += nj * float(valArr[2])
        mean = cumVal / cumN
        var = (cumSumSq - 2 * mean * cumVal + cumN * mean * mean) / cumN
        yield (mean, var)  # emit mean and var

    def steps(self):
        return ([
            self.mr(
                mapper=self.map,
                mapper_final=self.map_final,
                reducer=self.reduce,
            )
        ])


if __name__ == '__main__':
    MRmean.run()