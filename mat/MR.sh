#!/bin/bash

feat=feat/xit_bnf1.feat

pattern_type=merge
nHmms="50 100 300 500"
nState="3 5 7 9"
#e.g. given pattern/merge_50_3/ ...

head=merge
#generate mr_pattern/merge_50_3/ ...


python scripts/mfc.py $feat mfc/
mkdir -p exp/merge/
cd scripts
matlab -nodesktop -nosplash -nojvm -r "merge_bound('$pattern_type', [$nHmms], [$nState]); ver2_boundary2corpus('$pattern_type', [$nHmms], [$nState]); exit;"
cd ..

for i in $nHmms; do
    mkdir -p exp/${i}
    mkdir -p exp/${i}/result
    sh scripts/train_topic.sh $i
    cd scripts
    matlab -nodesktop -nosplash -nojvm -r "ver2_inference2mlf('$i'); exit;"
    cd ..

    python scripts/mlf2init.py exp/${i}/result/result.mlf init/${i}.txt

#    for j in $nState; do
#        ./scripts/zrst.sh $head $i $j
#    done
done
