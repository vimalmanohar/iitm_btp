#!/bin/bash

# This script is invoked from ../run.sh
# It contains some phoneTxCAT-related scripts that I am breaking out of the main run.sh for clarity.

. cmd.sh
. path.sh

train_cmd="queue.pl"
decode_cmd="queue.pl"
update_cmd="run.pl"

initSubStates=5200  
finalSubStates=5200

train_cond=clean
lm_suffix=bg_5k

num_iters=20
realign_iters='6 9 12 15'
stage=-10
gselect=50
full_gselect=15
nummixes=400
use_state_dep_map=false
use_weight_projection=true
user=$(whoami)
test_iter=$num_iters
num_transform_classes=1
num_cluster_weight_classes=1
train_scheme=5
use_full_covar=false
cluster_transforms_iters=5
use_sequential_transform_update=false
num_threads=12
use_block_diagonal_transform=false
use_diagonal_transform=false
use_block_diagonal_transform2=false
use_diagonal_transform2=false
use_block_diagonal_transform3=false
use_diagonal_transform3=false
roots_file=data/lang/phones/roots.int
use_class_dep_steps=false
recluster_iters=
reinitializeA=true
use_sequential_multiple_txclass_update=false

ubm_iters=3

prefix=WgtProj1

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;


#cdhmmName=tri1.$user.window3.3200.19200.$train_cond
cdhmmName=tri1.$user.htk1.delta3.6000.24000.$train_cond
#cdhmmName=tri2a.$user.htk1.delta3.6000.24000.$train_cond

if [ $use_full_covar == false ]; then
  ubm_iters=0
fi

diag=
if [ $ubm_iters == 0 ]; then
  $diag = "diag."
fi

ubmName=ubm.$user.${diag}delta3.$train_cond.$nummixes

realign_string=$(echo $realign_iters | sed 's/ /_/g')

phoneTxCATExptName=phoneTxCAT.$user.$prefix.tri1.delta333.6000.24000.${initSubStates}.${train_cond}.realign$realign_string.$nummixes

ubmDir=exp/ubm/$ubmName
cdhmmDir=exp/tri1/${cdhmmName}
phoneTxCATDir=exp/phoneTxCAT/$phoneTxCATExptName

mkdir -p $ubmDir
mkdir -p $phoneTxCATDir

decode_config=conf/decode.config
ubm_feat_config=conf/train_delta3.conf
phoneTxCAT_feat_config=conf/train_delta3.conf
final_config=conf/train_delta3.conf

NP_decode=66
NP_train=86

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
        steps/train_ubm.sh --silence-weight 0.5 --cmd "$train_cmd" --feat_config $ubm_feat_config $nummixes data/train_$train_cond data/lang ${cdhmmDir}_ali $ubmDir || exit 1;
    fi
else
    htkUBM=$ubmDir/MMF.nosil
    awk '/<NUMMIXES>/{start=1} {if (start) print} /<TRANSP>/{start=0}' $htkMMF | sed '/<TRANSP>/d' > $htkUBM
    ubm-copy --binary=false --htk-in $htkUBM $ubmDir/final.ubm || exit 1
fi

if [ $phoneTxCATTrainFlag == 1 ]; then

    bash -x steps/train_phoneTxCAT.sh --cmd "$train_cmd" \
        --feat_config $phoneTxCAT_feat_config --gselect $gselect --full-gselect $full_gselect --num_iters $num_iters \
        --realign_iters "$realign_iters" --stage $stage --use-state-dep-map $use_state_dep_map --use-weight-projection $use_weight_projection \
        --num-transform-classes $num_transform_classes --train-scheme $train_scheme --use-full-covar $use_full_covar --num-cluster-weight-classes $num_cluster_weight_classes \
        --num-threads $num_threads --cluster-transforms-iters $cluster_transforms_iters --use-sequential-transform-update $use_sequential_transform_update \
        --use-block-diagonal-transform $use_block_diagonal_transform --use-diagonal-transform $use_diagonal_transform \
        --use-block-diagonal-transform2 $use_block_diagonal_transform2 --use-diagonal-transform2 $use_diagonal_transform2 \
        --use-block-diagonal-transform3 $use_block_diagonal_transform3 --use-diagonal-transform3 $use_diagonal_transform3 \
        --roots_file $roots_file --use-class-dep-steps $use_class_dep_steps --use-sequential-multiple-txclass-update $use_sequential_multiple_txclass_update \
        --recluster_iters "$recluster_iters" --reinitializeA $reinitializeA \
    $initSubStates $finalSubStates \
    data/train_${train_cond} data/lang \
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

    utils/mkgraph.sh data/lang_test_$lm_suffix $phoneTxCATDir $phoneTxCATDir/graph_$lm_suffix || exit 1;

    for i in `find data -maxdepth 1 -name "test_clean_wv1" -type d`; do
        x=${i##*/}
        sh -x steps/decode_phoneTxCAT.sh --nj $NP_decode --gselect $gselect --full-gselect $full_gselect --cmd "$decode_cmd" \
        --feat_config $phoneTxCAT_feat_config \
        $phoneTxCATDir/graph_$lm_suffix $i $phoneTxCATDir/decode_${lm_suffix}_${x} || exit 1
        rm $decode_${lm_suffix}_${x}/gselect.*.gz &> /dev/null
        rm $decode_${lm_suffix}_${x}/lat.*.gz &> /dev/null

    done

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
