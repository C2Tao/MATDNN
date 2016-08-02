#!/bin/bash -xe

#eng: 12 target; xit: 16 target
exp=xit
#pattern_dir=pattern/$exp/${bnf}_${mr}/
bnf=2
mr=2
feat=feat/$exp/${exp}.mbf
ivector_ark=feat/$exp/ivector.ark
cmvn=true
normalizer=normalizer/$exp/${bnf}.txt

context=4
dir=exp/1st_${exp}
#context of feat + ivector; DNN input
full=$dir/full.feat

if [ "${exp}" == "xit" ]; then
    echo "xit" ;
    options=--silent
fi
mkdir -p $dir/caffe
mkdir -p $dir/h5

#concat feat with context and ivector to generate DNN input
python work/splice_feat.py --left-context 4 --right-context 4 --input-dim 39 --ivector-ark $ivector_ark $feat $full

#generate and apply cmvn
if $cmvn; then
    python work/normalize.py $full $normalizer
    python work/apply_cmvn.py $full $normalizer $dir/cmvn.feat
    full=$dir/cmvn.feat
fi

hdf5_dir=$dir/h5
pattern_dir=pattern/$exp/${bnf}_${mr}/*
pattern_list=$dir/pattern.list

# Generate pattern.list
realpath $pattern_dir/result/result.mlf > $pattern_list

# Generate training hdf5 file
python work/feat_pattern_to_h5.py --input-dim 751 $pattern_list $full $hdf5_dir $options

# Generate h5 list
realpath $hdf5_dir/* > $dir/train.list

#Caffe training
../caffe/build/tools/caffe train\
    -solver proto/solver/1st_${exp}_solver.prototxt \
    -iterations 200000 \
     | grep -v 'hdf5_data_layer'

##new bnf
#output_feat=$dir/${exp}_${bnf}_${mr}.feat
#model=$dir/caffe/model_iter_200000.caffemodel

##NN forward; generate bnf
#python work/deploy.py --gpu --output-layer ip3 --input-dim 751 --output-dim 39 $full $output_feat proto/deploy/1st_${exp}_deploy.prototxt $model

##result dir
#output_result=${exp}_${bnf}_${mr}_result
#mkdir -p $output_result
#cp $output_feat $output_result/
#cp $model $output_result/
#cp $normalizer $output_result/normalizer.txt
##track 1 evaluation
#timestep=work/${exp}_timesteps.txt
#output_feat_dir=$dir/output_${exp}_dir
#mkdir -p $output_feat_dir
#python work/feat2eval.py $output_feat $timestep $exp $output_feat_dir
#work/${exp}_eval1/eval1 -j 4 $output_feat_dir $output_result
