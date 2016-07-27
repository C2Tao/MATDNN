

import sys
import os
import argparse
parser = argparse.ArgumentParser(description='Splice feature in a context dependent way')
parser.add_argument('input_feat', metavar='input_feat', type=argparse.FileType('r'),
                   help='input feature file')
parser.add_argument('output_feat', metavar='output_feat', action='store', 
                   help='output feature file')
parser.add_argument('--left-context', dest='left_context', action='store',
                   default=4, type=int,
                   help='left context of the feat (default: 4)')
parser.add_argument('--right-context', dest='right_context', action='store',
                   default=4, type=int,
                   help='right context of the feat (default: 4)')
parser.add_argument('--input-dim', dest='feat_dim', action='store', type=int,
                   default=39,
                   help='dimension of the input feature (default: 39)')
parser.add_argument('--ivector-ark', dest='ivector_ark' , metavar='ivector_ark', action='store', 
                   default='' , help='ivector ark file')
args = parser.parse_args()

left_context = args.left_context
right_context = args.right_context
feat_dim = args.feat_dim
dim = left_context + right_context + 1
ivector_ark = args.ivector_ark
ivectors = {}
ivector_dim = 0
if ivector_ark:
    ivector_file = open( ivector_ark , 'r')
    for line in ivector_file:
	line = line.rstrip('\n')
	tokens = line.split()
	ivector = tokens[ 2:-1 ] 
	ivectors[ tokens[0] ] = ivector
	ivector_dim = len(ivector)
    ivector_file.close()

total_dim = dim * feat_dim + ivector_dim

output_feat = args.output_feat
output_feat_file = open( output_feat , 'w')
    

while True:
    line = args.input_feat.readline().rstrip('\n')
    if not line:
	break
    utt_id = line.rstrip('.wav') 
    #print utt_id
    output_feat_file.write(utt_id + '.wav\n' )

    feats = []
    frame_counter = 0
    line = args.input_feat.readline().rstrip('\n')
    while line:
        tokens = line.split()[3:] 
        feats.append( tokens )
        frame_counter += 1
        line = args.input_feat.readline().rstrip('\n')

    for i in range( frame_counter ):
	feat_str = '%04d %04d #' % ( i , i+1)
        for index in range(i - left_context , i + right_context + 1 ):
	    feat = None
            if index < 0:
	        feat =  feats[0] 
	    elif index >= frame_counter:
	        feat =  feats[frame_counter - 1 ] 
	    else:
	        feat =  feats[index] 
	    for f in feat:
	        feat_str += " " + f
        if ivector_dim != 0:
	    for ivec in ivectors[utt_id]:
		feat_str += " " + ivec

	feat_str += "\n" 
	output_feat_file.write( feat_str ) 
    
    output_feat_file.write( '\n' )


output_feat_file.close()
args.input_feat.close()
