#!/bin/bash

#. cmd.sh

train_cmd="queue.pl -V"
decode_cmd="queue.pl -V"

initSubStates=2000
finalSubStates=$initSubStates

prefix=dict2
stage=-15
nummixes=400
user=$(whoami)
num_iters=25
realign_iters="10 15 20"
test_iter=$num_iters

cdhmmName=tri1.$user.dict2.1800.9000

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

ubmName=ubm.$user.dict2.delta2.$nummixes

realign_string=$(echo $realign_iters | sed 's/ /_/g')
sgmmExptName=sgmm.$user.$prefix.$realign_string.${initSubStates}.${finalSubStates}.$nummixes

ubmDir=exp/ubm/$ubmName
cdhmmDir=exp/tri1/${cdhmmName}
sgmmDir=exp/sgmm/$sgmmExptName

mkdir -p $ubmDir
mkdir -p $sgmmDir

decode_config=conf/decode.config

NP_decode=20
NP_train=50

if (( $# == 0 )); then
    echo "Usage: sgmm_basic.sh <options>"
    echo "1 ubm Train"
    echo "2 sgmm Train"
    echo "3 sgmm Test"
    echo "5 final Variance Update"
    echo "--no-covar-update"
    echo "--htk-in"
    exit 0
fi

sgmmTrainFlag=0
finalVarUpdateFlag=0
sgmmTestFlag=0
ubmTrainFlag=0
noCovarUpdateFlag=0
htkUBMInFlag=0

while (( $# > 0 )); do
    case $1 in 
        1)
            ubmTrainFlag=1
            shift
            ;;
        2)
            sgmmTrainFlag=1
            shift
            ;;
        3)
            sgmmTestFlag=1
            shift
            ;;
        5)
            finalVarUpdateFlag=1
            shift
            ;;
        --no-covar-update)
            noCovarUpdateFlag=1
            shift
            ;;
        --htk-in)
            htkUBMInFlag=1
            shift
            ;;
        *)
            echo "Usage: sgmm_basic.sh <options>"
            echo "1 ubm Train"
            echo "2 sgmm Train"
            echo "3 sgmm Test"
            echo "5 final Variance Update"
            echo "--no-covar-update"
            echo "--htk-in"
            exit 1
    esac
done


## SGMM on top of Delta+Delta-Deltas features.
if [ $ubmTrainFlag == 1 ]; then
  steps/train_ubm.sh --silence-weight 0.5 --cmd "$train_cmd" 400 data/train data/lang ${cdhmmDir}_ali $ubmDir || exit 1;
fi

if [ $sgmmTrainFlag == 1 ]; then
  steps/train_sgmm.sh --stage $stage --num-iters $num_iters --realign-iters "$realign_iters" --cmd "$train_cmd" $initSubStates $finalSubStates data/train data/lang ${cdhmmDir}_ali $ubmDir/final.ubm $sgmmDir || exit 1;
fi

if [ $sgmmTestFlag == 1 ]; then
    
  [[ -f $sgmmDir/$test_iter.mdl ]] || exit 1
  cd $sgmmDir
  rm final.mdl 2> /dev/null
  ln -s $test_iter.mdl final.mdl || exit 1
  cd -

  utils/mkgraph.sh data/lang $sgmmDir $sgmmDir/graph || exit 1;

  steps/decode_sgmm.sh --config $decode_config --nj $NP_decode --cmd "$decode_cmd" \
  $sgmmDir/graph data/test $sgmmDir/decode || exit 1;

  steps/decode_sgmm.sh --config $decode_config --nj 10 --cmd "$decode_cmd" \
  $sgmmDir/graph data/test_feb89 $sgmmDir/decode_feb89 || exit 1;
fi

#steps/decode_sgmm.sh --use-fmllr true --config $decode_config --nj 20 --cmd "$decode_cmd" \
#  --transform-dir exp/tri3b/decode  $sgmmDir/graph data/test $sgmmDir/decode_fmllr || exit 1;
#
# #  Now we'll align the SGMM system to prepare for discriminative training.
# steps/align_sgmm.sh --nj 8 --cmd "$train_cmd" --transform-dir exp/tri3b \
#    --use-graphs true --use-gselect true data/train data/lang $sgmmDir $sgmmDir_ali || exit 1;
# steps/make_denlats_sgmm.sh --nj 8 --sub-split 20 --cmd "$decode_cmd" --transform-dir exp/tri3b \
#   data/train data/lang $sgmmDir_ali $sgmmDir_denlats
# steps/train_mmi_sgmm.sh --cmd "$decode_cmd" --transform-dir exp/tri3b --boost 0.2 \
#   data/train data/lang $sgmmDir_ali $sgmmDir_denlats $sgmmDir_mmi_b0.2 
#
# for iter in 1 2 3 4; do
#  steps/decode_sgmm_rescore.sh --cmd "$decode_cmd" --iter $iter \
#    --transform-dir exp/tri3b/decode data/lang data/test $sgmmDir/decode $sgmmDir_mmi_b0.2/decode_it$iter &
# done  
#
#wait 
#steps/decode_combine.sh data/test data/lang exp/tri1/decode exp/tri2a/decode exp/combine_1_2a/decode || exit 1;
#steps/decode_combine.sh data/test data/lang $sgmmDir/decode exp/tri3b_mmi/decode exp/combine_4a_3b/decode || exit 1;
## combining the sgmm run and the best MMI+fMMI run.
#steps/decode_combine.sh data/test data/lang $sgmmDir/decode exp/tri3b_fmmi_c/decode_it5 exp/combine_4a_3b_fmmic5/decode || exit 1;
#
#steps/decode_combine.sh data/test data/lang $sgmmDir_mmi_b0.2/decode_it4 exp/tri3b_fmmi_c/decode_it5 exp/combine_4a_mmi_3b_fmmic5/decode || exit 1;

