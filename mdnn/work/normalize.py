import sys
import numpy as np
from sklearn import preprocessing
nargv = len(sys.argv)
if nargv != 3:
    print 'Usage: python normalize.py feature normalizer'
    sys.exit()

feat1 = open( sys.argv[1], 'r' )
scaler = open( sys.argv[2], 'w' )

feats = []
frame_counter = 0
counter = 0

feat1.readline()
input_dim = len(feat1.readline().split()) - 3
feat1.close()
m = np.zeros(input_dim)
s = np.zeros(input_dim)

feat1 = open( sys.argv[1], 'r' )
for line in feat1:
    line = line.rstrip('\n') 
    if not line:
        continue
    tokens = line.split()
    if len(tokens) == 1:
        continue
    feat = [ float( i ) for i in tokens[3:] ]
    m += np.asarray(feat)
    counter += 1
feat1.close()
mean = m/counter

feat1 = open( sys.argv[1], 'r' )
for line in feat1:
    line = line.rstrip('\n') 
    if not line:
        continue

    tokens = line.split()
    if len(tokens) == 1:
        continue

    feat = [ float( i ) for i in tokens[3:] ]
    s += (np.asarray(feat) - mean)**2
feat1.close()
std = np.sqrt(s/counter)

for i in range(len(mean)):
    scaler.write(str(mean[i]) + ' ')
scaler.write('\n')
for i in range(len(std)):
    scaler.write(str(std[i]) + ' ')
scaler.write('\n')
