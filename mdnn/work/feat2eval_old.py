import os
import sys

nargv = len(sys.argv)
if nargv == 1:
    print 'Usage: python feat2eval1.py feature_filename eng/xit output_directory'
    print '   ex: python feat2eval1.py eng.mfc eng MFCC/'
    sys.exit()

if nargv != 4:
    print 'Error: # of input arguments mismatch.'
    print "Type 'python feat2eval1.py' to see its usage."
    sys.exit()


featfile = sys.argv[1]
out_dir = sys.argv[3]
exp = sys.argv[2]
wavName = ''
T = 0.0025


def getWaveName( exp , line):
    if exp == 'xit':
	return line.rstrip('\n')
    else:
	return line[:6]

for line in open(featfile, 'r'):
    if len(line) <= 1:
        continue

    if line[0] == 's' or line[0] == 'n':
        if (wavName != '') & ( getWaveName( exp , line ) != wavName):
            fout.close()
        
        if (wavName == '') | ( getWaveName( exp , line ) != wavName):
            wavName = getWaveName(exp , line)
            fout = open(os.path.join(out_dir, wavName + '.txt'), 'w')
            T = 0.0025

    else:
        T = T + 0.01
        parse = line.split(' ', 3)[3]
        fout.write('%.6f '%T + parse)



