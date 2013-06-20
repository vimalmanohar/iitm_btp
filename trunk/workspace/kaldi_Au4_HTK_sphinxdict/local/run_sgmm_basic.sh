#!/bin/bash

# This script is invoked from ../run.sh
# It contains some SGMM-related scripts that I am breaking out of the main run.sh for clarity.

. cmd.sh
. path.sh

train_cmd="queue.pl"
decode_cmd="queue.pl"

initSubStates=5200  
finalSubStates=5200

train_cond=clean
lm_suffix=bg_5k
num_iters=25

user=$(whoami)
realign_iters='10 15 20'
stage=-15
full_gselect=15
gselect=50
nummixes=400
cov_diag_ratio=2
ubm_iters=3

test_iter=$num_iters

prefix=

decode_config=conf/decode.config
ubm_feat_config=conf/train_delta3.conf
sgmm_feat_config=conf/train_delta3.conf
final_config=conf/train_delta3.conf

NP_decode=66
NP_train=86

# Note: you might want to try to give the option --spk-dep-weights=false to train_sgmm2.sh;
# this takes out the "symmetric SGMM" part which is not always helpful.

# SGMM system on si84 data [sgmm5a].  Note: the system we aligned from used the si284 data for
# training, but this shouldn't have much effect.

sgmmTrainFlag=0
finalVarUpdateFlag=0
sgmmTestFlag=0
ubmTrainFlag=0
noCovarUpdateFlag=0
htkUBMInFlag=0

if (( $# == 0 )); then
    echo "Usage: run_sgmm_basic.sh <options>"
    echo "1 ubm Train"
    echo "2 sgmm Train"
    echo "3 sgmm Test"
    echo "5 final Variance Update"
    echo "--no-covar-update"
    echo "--htk-in"
    exit 0
fi

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if (( $cov_diag_ratio > 1000 )); then
  ubm_iters=0
fi

diag=
if [ $ubm_iters == 0 ]; then
  diag="diag."
fi

#cdhmmName=tri1.vimal.window3.3200.19200.$train_cond
cdhmmName=tri1.$user.16mix1.delta3.6000.48000.$train_cond
#cdhmmName=tri2a.vimal.htk1.delta3.6000.24000.$train_cond

#ubmName=ubm.vimal.delta2.$train_cond
ubmName=ubm.$user.${diag}delta3.$train_cond.$nummixes
htkMMF=$HOME/scratch/Aurora4/exp/UBM/sgmm.vimal.natFeat.window.try1.nowindowUBM.windowSGMM.clean.400/MMF.nosil

realign_string=$(echo $realign_iters | sed 's/ /_/g')
#sgmmExptName=sgmm.vimal.cdhmm3.ubm3.sgmm3.3200.19200.${initSubStates}.${finalSubStates}.$train_cond
sgmmExptName=sgmm.$user.$prefix.${diag}delta333.6000.48000.${finalSubStates}.$train_cond.realign$realign_string.$nummixes
#sgmmExptName=sgmm.vimal.cdhmm3.ubm2.sgmm3.noVarUpdate.3600.21600.${initSubStates}.${finalSubStates}.$train_cond
#sgmmExptName=sgmm.vimal.cdhmm3.ubm2.sgmm3.noVarUpdate.3200.19200.${initSubStates}.${finalSubStates}.$train_cond

ubmDir=exp/ubm/$ubmName
cdhmmDir=exp/tri1/${cdhmmName}
sgmmDir=exp/sgmm/$sgmmExptName

mkdir -p $ubmDir
mkdir -p $sgmmDir

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
            echo "Usage: run_sgmm_basic.sh <options>"
            echo "1 ubm Train"
            echo "2 sgmm Train"
            echo "3 sgmm Test"
            echo "5 final Variance Update"
            echo "--no-covar-update"
            echo "--htk-in"
            exit 1
    esac
done

if [ $htkUBMInFlag == 0 ]; then
    if [ $ubmTrainFlag == 1 ]; then
        steps/train_ubm.sh --num-iters $ubm_iters --silence-weight 0.5 --cmd "$train_cmd" --feat_config $ubm_feat_config 400 data/train_$train_cond data/lang ${cdhmmDir}_ali $ubmDir || exit 1;
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

if [ $sgmmTrainFlag == 1 ]; then

    if [ $noCovarUpdateFlag == 1 ]; then

        sh steps/train_sgmm_novarupdate.sh --cmd "$train_cmd" \
          --feat_config $sgmm_feat_config --gselect $gselect --full-gselect $full_gselect --num_iters $num_iters \
          --realign_iters "$realign_iters" --stage $stage \
          --cov-diag-ratio $cov_diag_ratio --use-no-substates true \
            $initSubStates $finalSubStates \
            data/train_$train_cond data/lang \
            ${cdhmmDir}_ali $ubmDir/final.ubm $sgmmDir || exit 1;
    
    else
        
        bash -x steps/train_sgmm.sh --cmd "$train_cmd" \
          --feat_config $sgmm_feat_config --gselect $gselect --full-gselect $full_gselect --num_iters $num_iters \
          --realign_iters "$realign_iters" --stage $stage \
          --cov-diag-ratio $cov_diag_ratio --use-no-substates true \
            $initSubStates $finalSubStates \
            data/train_$train_cond data/lang \
            ${cdhmmDir}_ali $ubmDir/final.ubm $sgmmDir || exit 1;
  
        test_iter=$num_iters

    fi
    
fi

if [ $finalVarUpdateFlag == 1 ]; then
    
    dir=$sgmmDir.finalVarUpdate
    mkdir -p $dir                   # Make a new expt directory for final Variance Update

    # Some path requirement for Model update
    sdata=data/train_$train_cond/split$NP_train
    feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas --config=$final_config ark:- ark:- |"
    gselect_opt="--gselect=ark,s,cs:gunzip -c $sgmmDir/gselect.JOB.gz|"

    # Back-compatibility. Move the un-finalVarUpdated models back to their original files. 
    if [[ -L $sgmmDir/final.mdl.noUpdate ]]; then
        mv $sgmmDir/final.mdl.noUpdate $sgmmDir/final.mdl
        mv $sgmmDir/final.occs.noUpdate $sgmmDir/final.occs
    fi

    # Copy the model and occs from the expt dir to the new directory for final Variance update
    #cp $sgmmDir/final.mdl $dir/final.mdl
    #cp $sgmmDir/final.occs $dir/final.occs
  
    mkdir -p $dir/log
    cp $sgmmDir/final.mdl $dir/0.mdl
    
    rm $dir/tree &> /dev/null
    ln $sgmmDir/tree $dir/tree
    
    x=0
    niter=3

    while (( x < niter )); do

        utils/$train_cmd JOB=1:$NP_train $dir/log/acc.finalVarUpdate.JOB.log \
        sgmm-acc-stats --utt2spk=ark:$sdata/JOB/utt2spk \
        --update-flags=S "$gselect_opt" \
        --rand-prune=0.1 \
        $dir/$x.mdl "$feats" "ark,s,cs:gunzip -c $sgmmDir/ali.JOB.gz | ali-to-post ark:- ark:-|" \
        $dir/$x.JOB.acc || exit 1;

        utils/$train_cmd $dir/log/update.finalVarUpdate.log \
        sgmm-est --update-flags=S --split-substates=0 \
        --power=0.25 --write-occs=$dir/$((x+1)).occs \
        $dir/$x.mdl "sgmm-sum-accs - $dir/$x.*.acc|" \
        $dir/$((x+1)).mdl || exit 1;

        rm $dir/$x.mdl $dir/$x.*.acc $dir/$x.occs 2> /dev/null
        x=$((x+1))

    done 

    rm $dir/final.mdl $dir/final.occs 2> /dev/null
    ln -s $x.mdl $dir/final.mdl
    ln -s $x.occs $dir/final.occs
    
    #mv $dir/final.mdl $dir/final.mdl.noUpdate || exit 1
    #mv $dir/final.occs $dir/final.occs.noUpdate || exit 1
    
    #ln -s finalVarUpdate.mdl $dir/final.mdl
    #ln -s finalVarUpdate.occs $dir/final.occs
    
    utils/mkgraph.sh data/lang_test_$lm_suffix $dir $dir/graph_$lm_suffix || exit 1;

    for i in `find data -maxdepth 1 -name "test_*" -type d`; do
        x=${i##*/}
        steps/decode_sgmm.sh --nj $NP_decode --cmd "$decode_cmd" \
            --feat-config $final_config \
            $dir/graph_$lm_suffix $i $dir/decode_${lm_suffix}_${x} || exit 1
    done

    mkdir -p results/sgmm
    print_results $dir > results/sgmm/${dir##*/}

fi

if [ $sgmmTestFlag == 1 ]; then

    if [[ -L $sgmmDir/final.mdl.noUpdate ]]; then
        mv $sgmmDir/final.mdl.noUpdate $sgmmDir/final.mdl
        mv $sgmmDir/final.occs.noUpdate $sgmmDir/final.occs
    fi

    cd $sgmmDir
    rm final.mdl 2> /dev/null || exit 1
    ln -s $test_iter.mdl final.mdl
    cd -

    utils/mkgraph.sh data/lang_test_$lm_suffix $sgmmDir $sgmmDir/graph_$lm_suffix || exit 1;

    for i in `find data -maxdepth 1 -name "test_*" -type d`; do
        x=${i##*/}
        steps/decode_sgmm.sh --nj $NP_decode --cmd "$decode_cmd" \
        --feat_config $sgmm_feat_config --gselect $gselect --full-gselect $full_gselect \
        $sgmmDir/graph_$lm_suffix $i $sgmmDir/decode_${lm_suffix}_${x} || exit 1
        rm $decode_${lm_suffix}_${x}/gselect.*.gz &> /dev/null
        rm $decode_${lm_suffix}_${x}/lat.*.gz &> /dev/null

    done

    mkdir -p results/sgmm
    print_results $sgmmDir > results/sgmm/${sgmmDir##*/}
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