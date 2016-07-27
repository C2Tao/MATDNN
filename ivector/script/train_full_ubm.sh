#!/bin/bash -e

feat=`pwd`/feat
ubm=`pwd`/ubm
log=`pwd`/log
num_gselect=20 # cutoff for Gaussian-selection that we do once at the start.
iters=4
min_gaussian_weight=1.0e-04
remove_low_count_gaussians=true # set this to false if you need #gauss to stay fixed.
cleanup=true
# End configuration section.

if [ -f setup.sh ]; then . setup.sh; fi

mfcc_feats="ark:add-deltas scp:$feat/feat.scp ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- | select-voiced-frames ark:- scp:$feat/vad.scp ark:- |"

if [ -f $ubm/final.dubm ]; then # diagonal-covariance in $ubm
  gmm-global-to-fgmm $ubm/final.dubm $ubm/0.ubm 2> $log/to-fgmm.log;
elif [ -f $ubm/final.ubm ]; then
  cp $ubm/final.ubm $ubm/0.ubm;
else
  echo "ERROR: in $ubm, expecting final.ubm or final.dubm to exist"
  exit 1;
fi

echo "Gaussian selection (using diagonal form of model; selecting $num_gselect indices)"
gmm-gselect --n=$num_gselect "fgmm-global-to-gmm $ubm/0.ubm - |" "$mfcc_feats" \
  "ark:|gzip -c >$ubm/gselect.gz";

x=0

echo "Estimat FGMM"
while [ $x -lt $iters ]; do
  echo "Iter $x"
  fgmm-global-acc-stats "--gselect=ark,s,cs:gunzip -c $ubm/gselect.gz|" $ubm/$x.ubm "$mfcc_feats" \
	 $ubm/$x.acc;

  if [ $[$x+1] -eq $iters ];then
	 lowcount_opt="--remove-low-count-gaussians=true" # as specified by user.
  else
	 # On non-final iters, we in any case can't remove low-count Gaussians because it would
	 # cause the gselect info to become out of date.
	 lowcount_opt="--remove-low-count-gaussians=false"
  fi
  fgmm-global-est $lowcount_opt --min-gaussian-weight=$min_gaussian_weight --verbose=2 $ubm/$x.ubm "fgmm-global-sum-accs - $ubm/$x.acc |" \
	 $ubm/$[$x+1].ubm;
  rm $ubm/$x.acc $ubm/$x.ubm
  x=$[$x+1]
done

rm $ubm/gselect.gz

rm $ubm/final.ubm
mv $ubm/$x.ubm $ubm/final.ubm
