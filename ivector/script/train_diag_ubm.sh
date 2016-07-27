#!/bin/bash -e

ubm=`pwd`/ubm
feat=`pwd`/feat
log=`pwd`/log
threads="--num-threads=32"
#frames="--num-frames=500000"
min_gauss_weight="--min-gaussian-weight=0.0001"
gauss="--num-gauss=2048"
gauss_init="--num-gauss-init=1024"
iters="--num-iters=20"
mfcc_feats="ark:add-deltas scp:$feat/feat.scp ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- | select-voiced-frames ark:- scp:$feat/vad.scp ark:- |"

mkdir -p ubm

if [ -f setup.sh ]; then
  . setup.sh;
else
  echo "ERROR: setup.sh is missing!!";
  exit 1;
fi

#initial ubm from mfcc feature
echo "Initial diagonal UBM"
opts="$threads $frames $min_gauss_weight $gauss $gauss_init $iters"
gmm-global-init-from-feats $opts "$mfcc_feats" $ubm/0.dubm 2> $log/gmm_init.log

# Store Gaussian selection indices on disk-- this speeds up the training passes.
echo "Getting Gaussian-selection info"
gselect=30
iters=4
min_gauss_weight=0.0001 
gmm-gselect --n=$gselect $ubm/0.dubm "$mfcc_feats" "ark:|gzip -c >$ubm/gselect.gz" 2> $log/gselect.log

echo "Start Trainig dubm"
for x in `seq 0 $[$iters-1]`; do
  # Accumulate stats.
  echo "iter $x"
  gmm-global-acc-stats "--gselect=ark,s,cs:gunzip -c $ubm/gselect.gz|" \
	 $ubm/$x.dubm "$mfcc_feats" $ubm/$x.acc 2> $log/gmm-acc-stats.log;
  if [ $x -lt $[$iters-1] ]; then # Don't remove low-count Gaussians till last iter,
	 opt="--remove-low-count-gaussians=false" # or gselect info won't be valid any more.
  else
	 opt="--remove-low-count-gaussians=true"
  fi
  gmm-global-est $opt --min-gaussian-weight=$min_gauss_weight $ubm/$x.dubm "gmm-global-sum-accs - $ubm/$x.acc|" \
	 $ubm/$[$x+1].dubm 2>$log/gmm-est.log;
  rm $ubm/$x.acc $ubm/$x.dubm
done

rm $ubm/gselect.gz
mv $ubm/$iters.dubm $ubm/final.dubm || exit 1;

sec=$SECONDS
echo ""
echo "Execution time for whole script = `utility/timer.pl $sec`"
echo ""
