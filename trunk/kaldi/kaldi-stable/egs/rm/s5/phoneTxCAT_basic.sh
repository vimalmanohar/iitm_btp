#!/bin/bash

# This script is invoked from ../run.sh
# It contains some phoneTxCAT-related scripts that I am breaking out of the main run.sh for clarity.

. cmd.sh
. path.sh

#train_cmd="qparallel.pl -V"
#decode_cmd="qparallel.pl -V"

train_cmd="queue.pl -V"
decode_cmd="queue.pl -V"
update_cmd="run.pl"

initSubStates=2000
finalSubStates=$initSubStates

num_iters=30
realign_iters='10 15 20'
stage=-15
gselect=50
full_gselect=15
use_weight_projection=true
use_state_dep_map=false
nummixes=400
user=$(whoami)
test_iter=$num_iters
num_transform_classes=1
num_cluster_weight_classes=1
train_scheme=5
use_full_covar=true
cluster_transforms_iters=10
use_sequential_transform_update=false
num_threads=12
use_block_diagonal_transform=false
use_diagonal_transform=false
use_block_diagonal_transform2=false
use_diagonal_transform2=false
use_block_diagonal_transform3=false
use_diagonal_transform3=false

ubm_iters=3

prefix=WgtProj5

if (( $# == 0 )); then
    echo "Usage: phoneTxCAT_basic.sh <options>"
    echo "1 ubm Train"
    echo "2 phoneTxCAT Train"
    echo "3 phoneTxCAT Test"
    echo "5 final Variance Update"
    echo "--no-covar-update"
    echo "--htk-in"
    exit 0
fi

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

cdhmmName=tri1.$user.try3.1800.9000

if [ $use_full_covar == false ]; then
  ubm_iters=0
fi

diag=
if [ $ubm_iters == 0 ]; then
  $diag = "diag."
fi
ubmName=ubm.$user.${diag}delta2.$nummixes

realign_string=$(echo $realign_iters | sed 's/ /_/g')
#phoneTxCATExptName=phoneTxCAT.$user.tieWt2.tri2a.delta222.1800.9000.${initSubStates}.realign$realign_string.$num_iters
#phoneTxCATExptName=phoneTxCAT.$user.stateDep5.tri2a.delta222.1800.9000.${initSubStates}.realign$realign_string.$num_iters
#phoneTxCATExptName=phoneTxCAT.$user.stateDep7.tri1.delta222.1800.9000.${initSubStates}.realign$realign_string.$num_iters
#phoneTxCATExptName=phoneTxCAT.$user.WgtProj2.tri1.delta222.1800.9000.${initSubStates}.realign$realign_string.$num_iters
phoneTxCATExptName=phoneTxCAT.$user.$prefix.tri1.delta222.1800.9000.${initSubStates}.realign$realign_string.$nummixes
#phoneTxCATExptName=phoneTxCAT.$user.try6.tri1.cdhmm2.ubm2.phoneTxCAT2.1800.9000.${initSubStates}.realign$realign_string.$num_iters

ubmDir=exp/ubm/$ubmName
cdhmmDir=exp/tri1/${cdhmmName}
#cdhmmDir=exp/triDeltas/${cdhmmName}
phoneTxCATDir=exp/phoneTxCAT/$phoneTxCATExptName

mkdir -p $ubmDir
mkdir -p $phoneTxCATDir

decode_config=conf/decode.config
ubm_feat_config=conf/train_delta2.conf
phoneTxCAT_feat_config=conf/train_delta2.conf
final_config=conf/train_delta2.conf

NP_decode=20
NP_train=50

# Note: you might want to try to give the option --spk-dep-weights=false to train_sgmm2.sh;
# this takes out the "symmetric SGMM" part which is not always helpful.

# SGMM system on si84 data [sgmm5a].  Note: the system we aligned from used the si284 data for
# training, but this shouldn't have much effect.

phoneTxCATTrainFlag=0
finalVarUpdateFlag=0
phoneTxCATTestFlag=0
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
            phoneTxCATTrainFlag=1
            shift
            ;;
        3)
            phoneTxCATTestFlag=1
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
            echo "Usage: phoneTxCAT_basic.sh <options>"
            echo "1 ubm Train"
            echo "2 phoneTxCAT Train"
            echo "3 phoneTxCAT Test"
            echo "5 final Variance Update"
            echo "--no-covar-update"
            echo "--htk-in"
            exit 1
    esac
done

finalVarUpdateFlag=0
noCovarUpdateFlag=0
htkUBMInFlag=0

if [ $htkUBMInFlag == 0 ]; then
    if [ $ubmTrainFlag == 1 ]; then
        steps/train_ubm.sh --num-iters $ubm_iters --silence-weight 0.5 --cmd "$train_cmd" --feat_config $ubm_feat_config $nummixes data/train data/lang ${cdhmmDir}_ali $ubmDir || exit 1;
        htkUBM=$ubmDir/MMF.nosil
        echo "~o" > $htkUBM
        echo "<STREAMINFO> 1 39" >> $htkUBM
        echo "<VECSIZE> 39<NULLD><MFCC_D_A_Z_0><FULLC>" >> $htkUBM
        echo "~h \"SPEECH\"" >> $htkUBM
        echo "<BEGINHMM>" >> $htkUBM
        echo "<NUMSTATES> 3" >> $htkUBM
        echo "<STATE> 2" >> $htkUBM
        ubm-copy --binary=false --target-kind=htk $ubmDir/final.ubm - >> $htkUBM
        echo "<TRANSP> 3" >> $htkUBM
        echo "  0.0 1.0 0.0" >> $htkUBM
        echo "  0.0 0.999 0.001" >> $htkUBM
        echo "  0.0 0.0 0.0" >> $htkUBM
        echo "<ENDHMM>" >> $htkUBM
    fi
else
    htkUBM=$ubmDir/MMF.nosil
    awk '/<NUMMIXES>/{start=1} {if (start) print} /<TRANSP>/{start=0}' $htkMMF | sed '/<TRANSP>/d' > $htkUBM
    ubm-copy --binary=false --htk-in $htkUBM $ubmDir/final.ubm || exit 1
fi

if [ $phoneTxCATTrainFlag == 1 ]; then

    bash -x steps/train_phoneTxCAT.sh --cmd "$train_cmd" --update-cmd "$update_cmd" \
        --feat_config $phoneTxCAT_feat_config --gselect $gselect --full-gselect $full_gselect --num_iters $num_iters \
        --realign_iters "$realign_iters" --stage $stage --use-state-dep-map $use_state_dep_map --use-weight-projection $use_weight_projection \
        --num-transform-classes $num_transform_classes --train-scheme $train_scheme --use-full-covar $use_full_covar --num-cluster-weight-classes $num_cluster_weight_classes \
        --num-threads $num_threads --cluster-transforms-iters $cluster_transforms_iters --use-sequential-transform-update $use_sequential_transform_update \
        --use-block-diagonal-transform $use_block_diagonal_transform --use-diagonal-transform $use_diagonal_transform \
        --use-block-diagonal-transform2 $use_block_diagonal_transform2 --use-diagonal-transform2 $use_diagonal_transform2 \
        --use-block-diagonal-transform3 $use_block_diagonal_transform3 --use-diagonal-transform3 $use_diagonal_transform3 \
        $initSubStates $finalSubStates \
        data/train data/lang \
        ${cdhmmDir}_ali $ubmDir/final.ubm $phoneTxCATDir || exit 1;

    test_iter=$num_iters
    
fi


if [ $phoneTxCATTestFlag == 1 ]; then
    
    [[ -f $phoneTxCATDir/$test_iter.mdl ]] || exit 1
    cd $phoneTxCATDir
    rm final.mdl 2> /dev/null
    ln -s $test_iter.mdl final.mdl || exit 1
    cd -

    if [[ -L $phoneTxCATDir/final.mdl.noUpdate ]]; then
        mv $phoneTxCATDir/final.mdl.noUpdate $phoneTxCATDir/final.mdl
        mv $phoneTxCATDir/final.occs.noUpdate $phoneTxCATDir/final.occs
    fi

    #for i in test_feb89 test; do
    #    x=${i##*/}
    #    bash -x steps/decode_phoneTxCAT.sh --nj $NP_decode --cmd "$decode_cmd" \
    #    --feat_config $phoneTxCAT_feat_config --stage 3 \
    #    $phoneTxCATDir/graph $i $phoneTxCATDir/decode_${x} || exit 1
    #    rm $decode_${x}/gselect.*.gz &> /dev/null
    #    rm $decode_${x}/lat.*.gz &> /dev/null

    #done

    utils/mkgraph.sh data/lang $phoneTxCATDir $phoneTxCATDir/graph || exit 1;

    sh -x steps/decode_phoneTxCAT.sh --config $decode_config --gselect $gselect --full-gselect $full_gselect --nj $NP_decode --cmd "$decode_cmd" \
    $phoneTxCATDir/graph data/test $phoneTxCATDir/decode || exit 1;

    sh -x steps/decode_phoneTxCAT.sh --config $decode_config --gselect $gselect --full-gselect $full_gselect --nj $NP_decode --cmd "$decode_cmd" \
    $phoneTxCATDir/graph data/test_feb89 $phoneTxCATDir/decode_feb89 || exit 1;

    mkdir -p results/phoneTxCAT
fi

#(
#  steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
#    data/train_si84 data/lang exp/tri4b exp/tri4b_ali_si84 || exit 1;
#
#  steps/train_ubm.sh --cmd "$train_cmd" \
#    400 data/train_si84 data/lang exp/tri4b_ali_si84 exp/ubm5a || exit 1;
#
#  steps/train_sgmm2.sh --cmd "$train_cmd" \
#    7000 9000 data/train_si84 data/lang exp/tri4b_ali_si84 \
#    exp/ubm5a/final.ubm exp/sgmm2_5a || exit 1;
#
#  (
#    utils/mkgraph.sh data/lang_test_tgpr exp/sgmm2_5a exp/sgmm2_5a/graph_tgpr
#    steps/decode_sgmm2.sh --nj 10 --cmd "$decode_cmd" --transform-dir exp/tri4b/decode_tgpr_dev93 \
#      exp/sgmm2_5a/graph_tgpr data/test_dev93 exp/sgmm2_5a/decode_tgpr_dev93
#  ) &
#
#  steps/align_sgmm2.sh --nj 30 --cmd "$train_cmd" --transform-dir exp/tri4b_ali_si84 \
#    --use-graphs true --use-gselect true data/train_si84 data/lang exp/sgmm2_5a exp/sgmm2_5a_ali_si84 || exit 1;
#  steps/make_denlats_sgmm2.sh --nj 30 --sub-split 30 --cmd "$decode_cmd" --transform-dir exp/tri4b_ali_si84 \
#    data/train_si84 data/lang exp/sgmm2_5a_ali_si84 exp/sgmm2_5a_denlats_si84
#
#  steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" --transform-dir exp/tri4b_ali_si84 --boost 0.1 \
#    data/train_si84 data/lang exp/sgmm2_5a_ali_si84 exp/sgmm2_5a_denlats_si84 exp/sgmm2_5a_mmi_b0.1
#
#  for iter in 1 2 3 4; do
#    steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
#      --transform-dir exp/tri4b/decode_tgpr_dev93 data/lang_test_tgpr data/test_dev93 exp/sgmm2_5a/decode_tgpr_dev93 \
#      exp/sgmm2_5a_mmi_b0.1/decode_tgpr_dev93_it$iter &
#  done
#
#  steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" --transform-dir exp/tri4b_ali_si84 --boost 0.1 \
#   --update-opts "--cov-min-value=0.9" data/train_si84 data/lang exp/sgmm2_5a_ali_si84 exp/sgmm2_5a_denlats_si84 exp/sgmm2_5a_mmi_b0.1_m0.9
#
#  for iter in 1 2 3 4; do
#    steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
#      --transform-dir exp/tri4b/decode_tgpr_dev93 data/lang_test_tgpr data/test_dev93 exp/sgmm2_5a/decode_tgpr_dev93 \
#      exp/sgmm2_5a_mmi_b0.1_m0.9/decode_tgpr_dev93_it$iter &
#  done
#
#) &
#
#
#(
## The next commands are the same thing on all the si284 data.
#
## SGMM system on the si284 data [sgmm5b]
#  steps/train_ubm.sh --cmd "$train_cmd" \
#    600 data/train_si284 data/lang exp/tri4b_ali_si284 exp/ubm5b || exit 1;
#
#  steps/train_sgmm2.sh --cmd "$train_cmd" \
#   11000 25000 data/train_si284 data/lang exp/tri4b_ali_si284 \
#    exp/ubm5b/final.ubm exp/sgmm2_5b || exit 1;
#
#  (
#    utils/mkgraph.sh data/lang_test_tgpr exp/sgmm2_5b exp/sgmm2_5b/graph_tgpr
#    steps/decode_sgmm2.sh --nj 10 --cmd "$decode_cmd" --transform-dir exp/tri4b/decode_tgpr_dev93 \
#      exp/sgmm2_5b/graph_tgpr data/test_dev93 exp/sgmm2_5b/decode_tgpr_dev93
#    steps/decode_sgmm2.sh --nj 8 --cmd "$decode_cmd" --transform-dir exp/tri4b/decode_tgpr_eval92 \
#      exp/sgmm2_5b/graph_tgpr data/test_eval92 exp/sgmm2_5b/decode_tgpr_eval92
#
#    utils/mkgraph.sh data/lang_test_bd_tgpr exp/sgmm2_5b exp/sgmm2_5b/graph_bd_tgpr || exit 1;
#    steps/decode_sgmm2.sh --nj 10 --cmd "$decode_cmd" --transform-dir exp/tri4b/decode_bd_tgpr_dev93 \
#      exp/sgmm2_5b/graph_bd_tgpr data/test_dev93 exp/sgmm2_5b/decode_bd_tgpr_dev93
#    steps/decode_sgmm2.sh --nj 8 --cmd "$decode_cmd" --transform-dir exp/tri4b/decode_bd_tgpr_eval92 \
#      exp/sgmm2_5b/graph_bd_tgpr data/test_eval92 exp/sgmm2_5b/decode_bd_tgpr_eval92
#  ) &
#
#  steps/align_sgmm2.sh --nj 30 --cmd "$train_cmd" --transform-dir exp/tri4b_ali_si284 \
#    --use-graphs true --use-gselect true data/train_si284 data/lang exp/sgmm2_5b exp/sgmm2_5b_ali_si284 
#
#  steps/make_denlats_sgmm2.sh --nj 30 --sub-split 30 --cmd "$decode_cmd" --transform-dir exp/tri4b_ali_si284 \
#    data/train_si284 data/lang exp/sgmm2_5b_ali_si284 exp/sgmm2_5b_denlats_si284
#
#  steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" --transform-dir exp/tri4b_ali_si284 --boost 0.1 \
#    data/train_si284 data/lang exp/sgmm2_5b_ali_si284 exp/sgmm2_5b_denlats_si284 exp/sgmm2_5b_mmi_b0.1
#
#  for iter in 1 2 3 4; do
#    for test in eval92; do # dev93
#      steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
#        --transform-dir exp/tri4b/decode_bd_tgpr_${test} data/lang_test_bd_fg data/test_${test} exp/sgmm2_5b/decode_bd_tgpr_${test} \
#        exp/sgmm2_5b_mmi_b0.1/decode_bd_tgpr_${test}_it$iter &
#     done
#  done
#) &

wait
