feat=`pwd`/feat
log=`pwd`/log

mkdir -p feat
mkdir -p log

if [ -f 00.binary.path.sh ]; then . 00.binary.path.sh; else echo "ERROR: 00.binary.path.sh is missing!"; exit 1; fi
if [ -f 00.global.setting.sh ]; then . 00.global.setting.sh; else echo "ERROR: 00.global.setting.sh is missing!"; exit 1; fi

#####################################################################
#                                                                   #
# For most common setting , featbin = compute-mfcc-feats            #
#                                                                   #
#####################################################################

echo "Extracting wav set"
config=material/feat.conf
compute-mfcc-feats --verbose=2 --config=$config scp:material/wav.scp \
  ark,scp:$feat/feat.ark,$feat/feat.scp \
  2> $log/extract-mfcc.log

echo "Extracting valid scp for mfcc"
config=material/vad.conf
compute-vad --config=$config scp:$feat/feat.scp ark,scp:$feat/vad.ark,$feat/vad.scp 2> $log/extract-vad.log

sec=$SECONDS

echo ""
echo "Execution time for whole script = `utility/timer.pl $sec`"
echo ""
