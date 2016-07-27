import sys
from sys import argv
import asr
import util
# from zrst import pyASR
# from zrst import pySFX
# from zrst import pyEXT
# ######################phase1####################
# generate patterns

if len(argv) == 1:
    print 'Usage: python myResume.py [feature_dir] [feature_ext] [init_file] [#states] [output_dir]'
    sys.exit()

# ## set paths ###
corpus_path = 'UNUSED'
feature_path = argv[1]
feature_extension = argv[2]
#init generated by matlab/clusterDetection.m
initial_path = argv[3]
target_path = argv[5]
nState = int(argv[4])


features = []
for line in open(initial_path):
    if '.' in line:
        features.append(feature_path + line.split('.')[0] + feature_extension)
    

A = asr.ASR(corpus=corpus_path, 
    target=target_path, 
    nState=nState, 
    nFeature=39, 
    features = features, 
    dump = initial_path, 
    pad_silence=False)

#A.initialization('a')
A.readASR(target_path)

'''
A.iteration('a')
A.iteration('a_keep')
A.iteration('a_keep')
A.iteration('a_keep')

A.iteration('a_mix')
A.iteration('a_keep')
A.iteration('a_keep')
A.iteration('a_keep')
'''

A.iteration('a_keep')
A.iteration('a_mix')
A.iteration('a_keep')
A.iteration('a_keep')
A.iteration('a_keep')
A.iteration('a_keep')
A.iteration('a_keep')
A.iteration('a_keep')
A.iteration('a_keep')
A.iteration('a_keep')
A.iteration('a_keep')
A.iteration('a_keep')
A.iteration('a_keep')
A.iteration('a_keep')
A.iteration('a_keep')
