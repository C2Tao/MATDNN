#!/bin/sh

featPath=mfc/
feat_ext=.mfc
head=$1
nHmms=$2
nState=$3

init_dump=init/${nHmms}.txt
outputPath=mr_pattern/${head}_${nHmms}_${nState}/

python zrst/myEdit.py $featPath $feat_ext $init_dump $nState $outputPath
