import sys

mlf = open( sys.argv[1], 'r')
init = open( sys.argv[2], 'w')

mlf.readline()
for line in mlf:
    line = line.rstrip('\n')
    if line[0] == '.':
        continue
    if line[1] == '*':
        init.write(line[3:-4] + 'wav\n')
        continue
    tokens = line.split()
    init.write(tokens[2][1:] + '\n')
        
