import sys
import numpy as np
from sklearn import preprocessing
nargv = len(sys.argv)
if nargv != 4:
    print 'Usage: python apply_cmvn.py feature normalizer feature_cmvn'
    sys.exit()

feat1 = open( sys.argv[1], 'r' )
scaler = open( sys.argv[2], 'r' )
feat2 = open( sys.argv[3], 'w' )

utt_id = ''
feats = []
frame_counter = 0
count = 0

feat1.readline()
input_dim = len(feat1.readline().split()) - 3
feat1.close()

tokens = scaler.readline().split()
mean = np.asarray([ float(i) for i in tokens])
tokens = scaler.readline().split()
std = np.asarray([ float(i) for i in tokens])


feat1 = open( sys.argv[1], 'r' )
for line in feat1:
    line = line.rstrip('\n') 
    if not line:
        count += 1
        feat2.write( utt_id  + '.wav\n')
        feats_np = np.asarray(feats)
        cmvn = (feats_np - mean)/std
        for i in range( frame_counter):
            output_str = "%04d %04d #" % ( i, i+1 )
            for j in range( input_dim ):
                output_str += " " + str( cmvn[i, j] )
            output_str += '\n'
            feat2.write( output_str )
        feat2.write( '\n' )
        feats = []
        frame_counter = 0
        continue

    tokens = line.split()
    if len(tokens) == 1:
        utt_id = line.rstrip('.wav') 
        continue

    feat = [ float( i ) for i in tokens[3:] ]
    feats.append( feat )
    frame_counter += 1
feat1.close()
