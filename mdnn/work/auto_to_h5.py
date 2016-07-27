import sys
import h5py
import numpy as np
nargv = len(sys.argv)
if nargv == 1:
    print 'Usage: python auto_to_h5.py feature h5_dir/'
    sys.exit()
if nargv != 3:
    print "Error! Type 'python auto_to_h5.py' to see its usage."
    sys.exit()

feat1 = open( sys.argv[1], 'r' )
hdf_dir = sys.argv[2]


utt_id = ''
h5_fn = ''
feats = []
frame_counter = 0
count = 0
input_dim = 0

for line in feat1:
    line = line.rstrip('\n') 
    if not line:
        count += 1
        with h5py.File( h5_fn , 'w' ) as f:
            data = f.create_dataset( 'data' , ( frame_counter ,  input_dim , 1, 1 ) , maxshape=(None,input_dim,1,1) , chunks=True, dtype='float32' )
            for i in range( frame_counter):
                data[i , :, 0 , 0 ] = feats[i]
        feats = []
        frame_counter = 0
        continue



    tokens = line.split()
    if len(tokens) == 1:
        utt_id = line.rstrip('.wav') 
        h5_fn = hdf_dir + "/" + utt_id + ".h5"
        continue
    input_dim = len(tokens) - 3
    feat = [ float( i ) for i in tokens[3:] ]
    feats.append( feat )
    frame_counter += 1
feat1.close()
print count
print input_dim
    
    
