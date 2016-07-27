import sys
import numpy as np

sys.path.insert(0, 'zrst/')
from util import write_feature

feat = open(sys.argv[1],'r')
outfile = ''
for line in feat:
    line = line.rstrip('\n')
    if 'wav' in line:
        outfile = sys.argv[2] + '/' + line[:-3] + 'mfc'
        mfc = []
        continue
    if line == '':
        feature = np.asarray(mfc)
        write_feature(feature, outfile, period=100000)
    tokens = line.split()
    mfc.append([float(i) for i in tokens[3:]])
