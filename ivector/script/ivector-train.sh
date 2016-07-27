#!/bin/bash -e

if [ -f setup.sh ]; then
  . setup.sh;
else
  echo "ERROR: setup.sh is missing!";
  exit 1;
fi

dir=`pwd`/ivector
ubm=`pwd`/ubm
feat=`pwd`/feat
log_dir=$dir/log
mkdir -p $dir
mkdir -p $log_dir

fgmm-global-to-gmm $ubm/final.ubm $ubm/final.dubm

ivector_dim=400
use_weights=false
ivector-extractor-init --ivector-dim=$ivector_dim --use-weights=$use_weights \
  $ubm/final.ubm $dir/01.ie

num_gselect=20
min_post=0.025
mfcc_feat="cat $feat/feat.scp | add-deltas scp:- ark:- |"
( gmm-gselect --n=$num_gselect $ubm/final.dubm "ark:$mfcc_feat" ark:- \
    | fgmm-global-gselect-to-post --min-post=$min_post $ubm/final.ubm "ark:$mfcc_feat" \
        ark,s,cs:- ark:gpost.txt ) 2> $log_dir/gpost.log

num_samples_for_weights=3
num_iters=10
iter=1
while [ $iter -le $num_iters ]; do
  x=`printf "%02g" $iter`
  y=`printf "%02g" $[$iter+1]`
  log=$log_dir/acc.$x.log
  ivector-extractor-acc-stats --verbose=3 --num-samples-for-weights=$num_samples_for_weights \
    $dir/$x.ie "ark:$mfcc_feat" "ark,s,cs:gpost.txt" $dir/$x.acc 2> $log
  log=$log_dir/update.$x.log
  ivector-extractor-est --binary=false $dir/$x.ie $dir/$x.acc $dir/$y.ie 2> $log
  rm $dir/$x.ie $dir/$x.acc
  iter=$[$iter+1]
done

cp -f $dir/$y.ie $dir/final.ie
rm gpost.txt
