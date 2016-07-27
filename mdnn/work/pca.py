import sys
import numpy as np
from sklearn.decomposition import PCA
nargv = len(sys.argv)
if nargv == 1:
    print 'Usage: python pca.py input_dim output_dim feature_origin feature_cmvn'
    sys.exit()
if nargv != 5:
    print "Error! Type 'python cmvn.py' to see its usage."
    sys.exit()

input_dim = int(sys.argv[1])
output_dim = int(sys.argv[2])
feat1 = open( sys.argv[3], 'r' )
feat2 = open( sys.argv[4], 'w' )

utt_id = ''
feats = []
frame_counter = 0
count = 0
pca = PCA(n_components=output_dim)

for line in feat1:
    line = line.rstrip('\n') 
    if not line:
        count += 1
        feat2.write( utt_id  + '.wav\n')
        feats_np = np.asarray(feats)
        cmvn = pca.fit_transform(feats_np)
        for i in range( frame_counter):
            output_str = "%04d %04d #" % ( i, i+1 )
            for j in range( output_dim ):
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
print count
    
    
    