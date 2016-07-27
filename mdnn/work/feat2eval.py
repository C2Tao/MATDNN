import os
import sys

nargv = len(sys.argv)
if nargv == 1:
    print 'Usage: python feat2eval1.py feature_filename timesteps engORxit output_directory'
    print '   ex: python feat2eval1.py eng.mfc eng_timesteps.txt eng MFCC/'
    sys.exit()

if nargv != 5:
    print 'Error: # of input arguments mismatch.'
    print "Type 'python feat2eval1.py' to see its usage."
    sys.exit()


featfile = sys.argv[1]
timesteps = sys.argv[2]
lan = sys.argv[3]
out_dir = sys.argv[4]

if lan == 'eng':
    end = 6;
elif lan == 'xit':
    end = -5;
else:
    print "error: unknown language {}. Only 'eng' and 'xit' are valid.".format(lan)
    sys.exit()

wavName = ''
tin = open(timesteps, 'r')
for line in open(featfile, 'r'):
    enil = tin.readline()

    if len(line) <= 1:
        continue

    if (line[0] == 's') | (line[0] == 'n'):
        if (wavName != '') & (line[:end] != wavName):
            fout.close()
        
        if (wavName == '') | (line[:end] != wavName):
            wavName = line[:end]
            fout = open(os.path.join(out_dir, wavName + '.txt'), 'w')

    else:
        T = float(enil.split(' ')[4])
        parse = line.split(' ', 3)[3]
        fout.write('%.3f '%T + parse)

