
import sys
import os
import re
import h5py
import random
import argparse
import numpy as np
from sklearn import preprocessing
parser = argparse.ArgumentParser(description='Generate HDF5 file based on pattern list, feature list.')
parser.add_argument('pattern_list', metavar='pattern.list', type=argparse.FileType('r'),
                   help='pattern list contains the absolute path of .mlf file')
parser.add_argument('input_feat', metavar='input_feat', type=argparse.FileType('r'),
                   help='input feature file')
parser.add_argument('hdf_dir', metavar='hdf_dir', action='store', 
                   help='directory name of the hdf5 files')
parser.add_argument('--input-dim', dest='input_dim', action='store', type=int,
                   default=751,
                   help='dimension of the input feature (default: 751)')
parser.add_argument('--silent', dest='silent' ,action='store_true', 
                    help='flag it if the pattern contain silence and short pause')
args = parser.parse_args()
silent = 1
if args.silent :
    silent = 0
pattern_map = {}
pattern_count = 0
for line in args.pattern_list:
    pattern_mlf = line.rstrip('\n')
    pattern_file = open( pattern_mlf , 'r')
    utt_id = None
    frame_count = 0
    print pattern_count
    for pline in pattern_file:
	pline = pline.rstrip('\n')	
	if pline == "#!MLF!#":
	    continue
	if pline == ".":
	    frame_count = 0
	    utt_id = None
	    continue	
	tokens = pline.split()
	if len(tokens) == 1:
	    utt_id = pline[3:-5]
	    #print utt_id
	    if utt_id not in pattern_map:
                pattern_map[ utt_id ] = []
	    pattern_map[utt_id ].append( [] )
	    continue
	pattern = silent
	if tokens[2] != 'sil' and tokens[2] != 'sp':
	    pattern = int( tokens[2][1:]) - silent
	end = int( tokens[1] ) / 100000
	while frame_count < end:
	    frame_count += 1
	    pattern_map[ utt_id ][ pattern_count ].append( pattern )
    pattern_count += 1
    pattern_file.close() 
   
args.pattern_list.close()
# ensure dir
if not os.path.exists(args.hdf_dir):
    os.makedirs( args.hdf_dir )


input_dim = args.input_dim
   
utt_id = ''
utt_map = None
h5_fn = ''
feats = []
frame_counter = 0
neglect_utt = False 

for line in args.input_feat:
    line = line.rstrip('\n') 
    if not line:
	if neglect_utt:
	    neglect_utt = False
	    continue
	with h5py.File( h5_fn , 'w' ) as f:
	    data = f.create_dataset( 'data' , ( frame_counter ,  input_dim , 1, 1 ) , maxshape=(None,input_dim,1,1) , chunks=True, dtype='float32' )
	    label = f.create_dataset( 'label' , ( frame_counter , pattern_count ) , maxshape=(None, pattern_count) , dtype='float32' , chunks=True)
	    for i in range( frame_counter):
		data[i , :, 0 , 0 ] = feats[i]
	        label_vec = []
	        for j in range( pattern_count ):
		    pattern = utt_map[j][i]
	            label_vec.append( pattern )
		label[i , : ] = label_vec
	feats = []
	frame_counter = 0
	continue

    if neglect_utt:
	continue

    tokens = line.split()
    if len(tokens) == 1:
	utt_id = line.rstrip('.wav')
        #print utt_id
        if utt_id not in pattern_map:
	    print "utt_id not found in pattern map."
	    neglect_utt = True
	    continue
        utt_map = pattern_map[utt_id] 
	h5_fn = args.hdf_dir + "/" + utt_id + ".h5"
	continue
    #print line

    feat = [ float( i ) for i in tokens[3:] ]
    feats.append( feat )
    frame_counter += 1

args.input_feat.close()
