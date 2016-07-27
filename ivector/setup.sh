
work_root=/home/coldsheep/microsoft/Research/muli-dnn-adapt
htk_root=/home/m1010/Tools/htk
openfst_root=/home/m1010/Tools/kaldi-trunk/tools/openfst
kaldi_root=/home/m1010/Tools/kaldi-trunk/src
vulcan_root=/home/m1010/Tools/vulcan

PATH=$htk_root/HTKTools:$PATH
PATH=$htk_root/HTKLVRec:$PATH
PATH=$openfst_root/bin:$PATH
PATH=$kaldi_root/bin:$PATH
PATH=$kaldi_root/fstbin/:$PATH
PATH=$kaldi_root/gmmbin/:$PATH
PATH=$kaldi_root/featbin/:$PATH
PATH=$kaldi_root/sgmmbin/:$PATH
PATH=$kaldi_root/sgmm2bin/:$PATH
PATH=$kaldi_root/fgmmbin/:$PATH
PATH=$kaldi_root/latbin/:$PATH
PATH=$kaldi_root/nnetbin/:$PATH
PATH=$kaldi_root/ivectorbin/:$PATH
PATH=$vulcan_root/bin/:$PATH
PATH=$vulcan_root/HDecode++/:$PATH
export PATH=$PATH

##

export mfcc_all="cat material/feat.scp | add-deltas scp:- ark:- |"
export mfcc_train="cat material/train.scp | add-deltas scp:- ark:- |"
export mfcc_train_shuffled="cat material/train.scp | utility/shuffle_list.pl 777 | add-deltas scp:- ark:- |"
export mfcc_dev="cat material/dev.scp | add-deltas scp:- ark:- |"

export ubm_feat_train="cat material/train.scp | add-deltas scp:- ark:- |"
