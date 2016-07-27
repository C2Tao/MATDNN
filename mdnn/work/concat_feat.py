
import sys
import os
import argparse
parser = argparse.ArgumentParser(description='Generate HDF5 file based on pattern list, feature list.')
parser.add_argument('feat_file', metavar='feat', nargs='+', type=argparse.FileType('r'),
                   help='feature files that are to caoncat')
parser.add_argument('--output', dest='output' ,action='store', 
                    help='output file. default stdout' , default='')
args = parser.parse_args()

outfile = sys.stdout

if args.output != '':
    outfile = open( args.output , 'w')

#print args.feat_file

is_eof = False
while True:
    feats = []
    feat_id = None
    for feat_file in args.feat_file:
	line = feat_file.readline().rstrip('.wav\n')
	if not line:
	    is_eof = True
	    break
	if feat_id != None and feat_id != line:
	    raise Exception("feat_id mismatched")
	else:
	   feat_id = line
	    
	frame_counter = 0
	line = feat_file.readline().rstrip('\n')
        while line:
	    tokens = line.split()[3:]
	    if len( feats ) <= frame_counter:
		feats.append( [] )
	    feats[ frame_counter ].extend( tokens )
            frame_counter += 1
	    line = feat_file.readline().rstrip('\n')


    if is_eof:
	break

    outfile.write("%s.wav\n" % feat_id )
    for i in range( len(feats) ):
        out_str = "%04d %04d #" % ( i , i+1 )
	for f in feats[i]:
	    out_str = out_str + " %s" % f
	out_str = out_str + "\n"
        outfile.write( out_str )

    outfile.write('\n')
	        

        

if args.output != '':
    outfile.close()
