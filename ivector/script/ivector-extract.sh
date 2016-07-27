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
config=material/vad.conf
mkdir -p $dir
mkdir -p $log_dir

cp material/wav.scp $dir/extract.wav.scp
cat $dir/extract.wav.scp | compute-mfcc-feats scp:- ark:- | compute-vad --config=$config ark:- ark,scp:$dir/vad.ark,$dir/vad.scp

cp $feat/feat.scp $dir/extract.scp
scp=$dir/extract.scp
compute-vad --config=$config scp:$scp ark,t,scp:$dir/vad.ark,$dir/vad.scp

mfcc_feat="cat $dir/extract.scp | add-deltas scp:- ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- | select-voiced-frames ark:- scp,s,cs:$dir/vad.scp ark:- |"
num_gselect=20
min_post=0.025
gmm-gselect --n=$num_gselect $ubm/final.dubm "ark:$mfcc_feat" ark:- \
  | fgmm-global-gselect-to-post --min-post=$min_post $ubm/final.ubm "ark:$mfcc_feat" \
      ark,s,cs:- ark:- \
  | ivector-extract --verbose=2 $dir/final.ie "ark:$mfcc_feat" ark,s,cs:- \
      ark,scp,t:$dir/ivector.ark,$dir/ivector.scp
