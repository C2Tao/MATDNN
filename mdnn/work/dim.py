import sys
nargv = len(sys.argv)
if nargv == 1:
    print 'Usage: python dim.py feature'
    sys.exit()

feat = open( sys.argv[1], 'r' )
feat.readline()
token = feat.readline().split()
print len(token) - 3
