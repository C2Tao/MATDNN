from sys import argv
from random import randint

x = argv[1]
remove = argv[2] == 'r'
gen = int(argv[3])
I = open(x)
prev = ''
if remove:
	O = open(x[:-4]+'_flt_rmv.txt','w')
elif gen:
	O = open(x[:-4]+'_flt_gen.txt','w')
else:
	O = open(x[:-4]+'_flt.txt','w')

if remove:
	for line in I:
		if 'wav' in line:
			prev = line

		else:
			if prev:
				O.write(prev)
				prev = ''

			for p in line.strip('\n').split():
				if p: O.write(p+'\n')

elif gen:
	for line in I:
		if 'wav' in line:
			if prev:
				O.write(prev)
				O.write('%d\n'%randint(1, gen))
			prev = line

		else:
			if prev:
				O.write(prev)
				prev = ''
			for p in line.strip('\n').split():
				if p: O.write(p+'\n')

else:
	for line in I:
		if 'wav' in line:
			O.write(line)

		else:
			for p in line.strip('\n').split():
				if p: O.write(p+'\n')
