#######################################################################
#                                                                     #
# This script is written by                                           #
#   Ching-Feng Yeh @ National Taiwan University Speech Lab            #
#                                                                     #
# For bug report or contact :                                         #
#   andrew.yeh.1987@gmail.com                                         #
#                                                                     #
# These scripts are modified based on Kaldi example scripts ,         #
#   and several perl/python programs are directly copied.             #
#                                                                     #
# For commercial use , please check the license of Kaldi              #
#                                                                     #
#######################################################################

#######################################################################
#                                                                     #
# Since symbols for silence phonemes differ from corpus to corpus ,   #
#   and are essential for selecting phoneme HMM prototype ,           #
#   you should specify $silphones here.                               #
# For example , for TIMIT settings ,                                  #
#   silphones="cl epi sil vcl" is adequate.                           #
#                                                                     #
#######################################################################

silphones="cl epi sil vcl"
sil=sil

#######################################################################
#                                                                     #
# $featbin is the kaldi tool for feature extaction .                  #
#                                                                     #
# For example ,                                                       #
#   if $featbin = [ compute-mfcc-feats ] ,                            #
#     then mel-scale frequency ceptral coefficients is adopted .      #
#   if $featbin = [ compute-plp-feats ] ,                             #
#     then perceptual linear predictive is adopted                    #
#                                                                     #
# Currently known tools provided by Kaldi :                           #
#   (1) compute-mfcc-feats                                            #
#   (2) compute-plp-feats                                             #
#   (3) compute-fbank-feats                                           #
#   (4) compute-spectrogram-feats                                     #
#                                                                     #
#######################################################################

export featbin=compute-mfcc-feats

#######################################################################
#                                                                     #
# $feats_* are feature specifications in both training and decoding . #
#                                                                     #
# Taking training set specification for example ,                     #
#   if speaker-level cmvn is adopted ,                                #
#     feats_tr="ark:compute-cmvn-stats \                              #
#                     --spk2utt=ark:material/train.spk2utt \          #
#                     scp:feat/feat.train.scp ark:- \                 #
#                 | apply-cmvn --norm-vars=true \                     #
#                     --utt2spk=ark:material/train.utt2spk \          #
#                     ark:- scp:feat/feat.train.scp ark:- \           #
#                 | add-deltas --delta-order=2 \                      #
#                     ark:- ark:- |"                                  #
#     , in which cmvn statistics are computed and applied for each    #
#     speaker, with delta- and delta-delta- coefficients appended     #
#     in last step , is a correct specification .                     #
#   ( attention , for per-speaker level , *.spk2utt and *.utt2spk     #
#     files is essential and you should check at the beginning )      #
#                                                                     #
# The above example looks messy , so here is another simple one ,     #
#   if ordinary 39-dim mfcc is desired ,                              #
#     feats_tr="ark:add-deltas scp:feat/feat.train.scp ark:-|"        #
#   is a correct specification .                                      #
#                                                                     #
# Here all sets including train/dev/test should be specified          #
#                                                                     #
# Additionally, $nn_feats_* are feature specifications for neural     #
#   network training , which is original features splices of          #
#   consecutive frames .                                              #
# $context_left and $context_right are the numbers of                 #
#   consecutive frames to be splice , respectively.                   #
#                                                                     #
#######################################################################

export feat_dim=39

export context_left=4
export context_right=4

export scp_tr="feat/feat.train.scp"
export feats_tr="ark:compute-cmvn-stats --spk2utt=ark:material/train.spk2utt scp:$scp_tr ark:- \
                 | apply-cmvn --norm-vars=false --utt2spk=ark:material/train.utt2spk ark:- scp:$scp_tr ark:- \
                 | add-deltas ark:- ark:- |"
export nn_feats_tr="$feats_tr splice-feats --print-args=false \
                    --left-context=$context_left --right-context=$context_right ark:- ark:- |"

export scp_dv="feat/feat.dev.scp"
export feats_dv="ark:compute-cmvn-stats --spk2utt=ark:material/dev.spk2utt scp:$scp_dv ark:- \
                 | apply-cmvn --norm-vars=false --utt2spk=ark:material/dev.utt2spk ark:- scp:$scp_dv ark:- \
                 | add-deltas ark:- ark:- |"
export nn_feats_dv="$feats_dv splice-feats --print-args=false \
                    --left-context=$context_left --right-context=$context_right ark:- ark:- |"

export scp_ts="feat/feat.test.scp"
export feats_ts="ark:compute-cmvn-stats --spk2utt=ark:material/test.spk2utt scp:$scp_ts ark:- \
                 | apply-cmvn --norm-vars=false --utt2spk=ark:material/test.utt2spk ark:- scp:$scp_ts ark:- \
                 | add-deltas ark:- ark:- |"
export nn_feats_ts="$feats_ts splice-feats --print-args=false \
                    --left-context=$context_left --right-context=$context_right ark:- ark:- |"

#######################################################################
#                                                                     # 
# Here is the specification of labels for training                    #
#   and developing ( or cross-validation ) .                          #
# $oov_opt indicates the way to deal with OOVs for word table         #
# Since Kaldi accepts only integer labels during processing ,         #
#   text labels are transformed to integers.                          #
#                                                                     #
#######################################################################

export oov_opt="--ignore-oov"

export scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"

export label_tr="ark:utils/sym2int.pl $oov_opt --ignore-first-field train/words.txt < material/train.text |"

export label_dv="ark:utils/sym2int.pl $oov_opt --ignore-first-field train/words.txt < material/dev.text |"

#######################################################################
#                                                                     #
# The way to normalize transcriptions differs from corpus to corpus.  #
# For example , in most Mandarin systems , word-level transcription   #
#   is generated with decoding process ( WFST or Viterbi ) , but      #
#   character-level accuracy is desired .                             #
# Therefore , the recognized result and corresponding groud truth both#
#   require further normalization .                                   #
#                                                                     #
# For another example , in TIMIT common settings , 48 phonemes are    #
#   used in training and 39 phonemes is used for testing ,            #
#   which means the phonemes in the recognized result require mapping #
#   and this can be done by following specifiactions .                #
#                                                                     #
# $norm_trans indicates the method of normalizing transcriptions .    #
# $trans_* are corresponding specification of transcriptions for      #
#   each set.                                                         #
#                                                                     #
#######################################################################

export norm_trans="utils/int2sym.pl --ignore-first-field train/words.txt \
                   | utils/timit_norm_trans.pl -i - -m material/phones.60-48-39.map -from 48 -to 39"

export trans_dv="ark:material/dev.text"

export trans_ts="ark:material/test.text"

#######################################################################
#                                                                     #
# In DNN and MLP training , randomized initialization is required .   #
#                                                                     #
# However , randomized initialization usually leads to experimental   #
#   results which are not reproducable and hard to debug with .       #
#                                                                     #
# Therefore , in this script set , you can specify seeds for the      #
#   random number generator to make the experimental results stable . #
#                                                                     #
# It is acceptable if these seeds are not specified .                 #
# In such condition , random seeds will be adopted , and these seeds  #
#   will be stored in "seeds.txt" in corresponding experimental       #
#   directory .                                                       #
#                                                                     #
#######################################################################

#export rbm_seeds=(14919 1986 20726 6672 19095 14444 16850 15036 8259 27785)
export rbm_seeds=()
#export ft_seeds=(3891 25756 316 18647 6409 29901 19240 14402 19990 29265)
export ft_seeds=()
#export mlp_seeds=(21299 20141 17788 27708 26710)
export mlp_seeds=()

