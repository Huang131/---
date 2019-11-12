from mrjob.protocol import JSONProtocol
import numpy as np

fw = open(r'15-MapReduce\kickStart.txt', 'w')
for i in [1]:
    for j in range(100):
        fw.write('["x", %d]\n' % np.random.randint(200))
fw.close()