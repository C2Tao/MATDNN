import sys
nargv = len(sys.argv)
if nargv == 1:
    print 'Usage: python feat_combine.py feature_one feature_two output_feat'
    sys.exit()
if nargv != 4:
    print "Error! Type 'python feat_combine.py' to see its usage."
    sys.exit()

feat1 = open( sys.argv[1], 'r' )
feat2 = open( sys.argv[2], 'r' )
fout =  open( sys.argv[3], 'w' )

for line in feat1:
    token = feat2.readline().split()
    if len(token) <= 1:
        fout.write(line)
        continue
    fout.write( line.strip('\n') );
    for s in token[3:]:
        fout.write( ' ' + s )
    fout.write('\n')

