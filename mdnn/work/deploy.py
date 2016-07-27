#!/usr/bin/env python
import numpy as np
import os
import sys
import argparse
import glob
import time

sys.path.insert(0, '../caffe/python')
import caffe


def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="a file containing original features"
    )
    parser.add_argument(
        "output_file",
        help="a file that will store all the features"
    )

    parser.add_argument(
        "prototxt",
        help="Model definition file."
    )
    parser.add_argument(
        "caffemodel",
        help="Trained model weights file."
    )
    # Optional arguments.
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--input-dim",
        default=751,
	dest='input_dim',
	type=int,
        help="input dimension of original feature. default:751"
    )
    parser.add_argument(
        "--output-dim",
        default=39,
	dest='output_dim',
	type=int,
        help="output dimension of original feature. default:39"
    )
    parser.add_argument(
        "--output-layer",
        default='ip3',
	dest='output_layer',
        help="name of output layer. default:ip3"
    )
 

    args = parser.parse_args()

    input_dim = args.input_dim
    output_dim = args.output_dim
    if args.gpu:
        print 'GPU mode'
	caffe.set_mode_gpu()
    else:
	caffe.set_mode_cpu()


    input_file  = open( args.input_file ,   'r')
    output_file = open( args.output_file  , 'w')
    # Make net.
    net   = caffe.Classifier(args.prototxt, args.caffemodel)


    # Load numpy array (.npy), directory glob (*.jpg), or image file.

    
    start = time.time()
    while True:
	utt_id = ''
	frame_counter = 0
	line = input_file.readline().rstrip('\n')
	if not line:
	    break
	feats = []
	while line:
	    tokens = line.split()
	    if len(tokens) == 1:
		utt_id = line.rstrip('.wav')
		feats = []
		frame_counter = 0
	    else:
		feats.append( [float(f) for f in tokens[3:] ] )
		frame_counter += 1		

	    line = input_file.readline().rstrip('\n')

	#print frame_counter
	#print input_dim
	
	input_batch = np.ndarray( dtype='float32' , shape=(frame_counter,  input_dim ,1 , 1) )
	for i in range(frame_counter):
	    input_batch[ i , :, 0 , 0] = feats[i][:]
	print utt_id
	net.blobs['data'].reshape( frame_counter ,  input_dim, 1,1 )
	output = net.forward( data=input_batch )
	output_feats = np.ndarray( dtype='float32' , shape=( frame_counter , output_dim ) )
	output_feats[ : , : ] = output[ args.output_layer ][: , :  ] 

	output_file.write( utt_id  + '.wav\n')
	for i in range( frame_counter ):
	    output_str = "%04d %04d #" % ( i , i+1 )
	    for j in range( output_dim ):
		output_str += " " + str( output_feats[i , j])
	    output_str += '\n'
	    output_file.write( output_str )
	    #print output_str
	output_file.write( '\n' )

    input_file.close()
    output_file.close()
    # Classify.
    print "Done in %.2f s." % (time.time() - start)

    # Save
    #np.save(args.output_file, predictions)


if __name__ == '__main__':
    main(sys.argv)
