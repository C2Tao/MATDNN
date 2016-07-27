#!/bin/bash -xe

#eng: 12 target; xit: 16 target
exp=xit
feat=feat/${exp}/${exp}.mbf
train_result=xit_2_2_result/
normalizer=$train_result/normalizer.txt
model=$train_result/model_iter_200000.caffemodel

cmvn=true
ivector_ark=feat/${exp}/ivector.ark

dir=exp_test/
context=4
#context of feat + ivector; DNN input
full=$dir/full.feat

#concat feat with context and ivector to generate DNN input
python work/splice_feat.py --left-context 4 --right-context 4 --input-dim 39 --ivector-ark $ivector_ark $feat $full

#apply cmvn
if $cmvn; then
    python work/apply_cmvn.py $full $normalizer $dir/cmvn.feat
    full=$dir/cmvn.feat
fi

#new bnf
output_feat=$dir/output.feat

#NN forward; extract bnf
python work/deploy.py --gpu --output-layer ip3 --input-dim 751 --output-dim 39 $full $output_feat proto/deploy/1st_${exp}_deploy.prototxt $model

#result dir
output_result=test_result/
mkdir -p $output_result
cp $output_feat $output_result/

##track 1 evaluation
#timestep=work/${exp}_timesteps.txt
#output_feat_dir=$dir/output_${exp}_dir
#mkdir -p $output_feat_dir
#python work/feat2eval.py $output_feat $timestep $exp $output_feat_dir
#work/${exp}_eval1/eval1 -j 4 $output_feat_dir $output_result

