#!/bin/bash

#. ./cmd.sh || exit 1 ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

train_cmd="queue.pl"
decode_cmd="queue.pl"
run_cmd="queue.pl"

PATH_BASE=`pwd`

user=$(whoami)

MAIN_DIR=/speech/$user/workspace/kaldi_au4_HTK_sphinxdict
TEMP_DIR=/speech/$user/scratch/kaldi_au4_HTK_sphinxdict

DATA_DIR=$TEMP_DIR/data
data=$PATH_BASE/data

au4=/speech/$user/Database/aurora4/speech_data_aurora4
wsj0=/speech/$user/Database/WSJ284/WSJ0-84
wsj_lmdir=$wsj0/WSJ0/LNG_MODL/base_lm

use_htk_features=true
au4=/speech/$user/scratch/Aurora4/Features/feat_orig

NP_feat=50
NP_train=86
NP_decode=66

train_cond=clean
lm_suffix=bg_5k

initGaussians=6000         # Number of tied states
finalGaussians=24000        # Total number of Gaussian mixtures

initGaussiansDeltas=$initGaussians
finalGaussiansDeltas=$finalGaussians

prefix=try2
stage=-6
num_iters=40

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

monoName=mono.$user.$prefix.delta3.$train_cond 
tri1Name=tri1.$user.$prefix.$initGaussians.$finalGaussians.$train_cond
triDeltasName=tri2a.$user.$prefix.$initGaussiansDeltas.$finalGaussiansDeltas.$train_cond

monoDir=exp/mono/$monoName
tri1Dir=exp/tri1/$tri1Name
triDeltasDir=exp/tri2a/$triDeltasName

decode_config=conf/decode.config
feat_config=conf/train_delta3.conf
mfcc_config=conf/htk.conf

mfccdir=$TEMP_DIR/features_htk

rm -f exp 2> /dev/null
mkdir -p $TEMP_DIR/exp || exit 1
ln -s $TEMP_DIR/exp . || exit 1

mkdir -p $data/lang

mkdir -p exp/mono
mkdir -p exp/tri1
mkdir -p exp/tri2a

mkdir -p $monoDir

if (( $# == 0 )); then

    echo "Usage: run_basic.sh <options>"
    echo "1 Prepare Data"
    echo "2 Prepare Language Model"
    echo "3 Extract MFCC"
    echo "4 Monophone train"
    echo "5 Triphone train and test"
    echo "6 UBM train"
    echo "7 SGMM train"
    echo "8 SGMM test"
    echo "9 Deltas train and test"
    echo "10 Final Variance Update"
    echo "20 Force"
    exit 0
fi

prepareDataFlag=0
prepareLangFlag=0
extractFeaturesFlag=0
monoTrainFlag=0
triTrainFlag=0
triTestFlag=0
ubmTrainFlag=0
sgmmTrainFlag=0
sgmmTestFlag=0
deltasTrainFlag=0
finalVarUpdateFlag=0
noCovarUpdateFlag=0
force=0

while (( $# > 0 )); do
    case $1 in 
        1)
            prepareDataFlag=1
            shift
            ;;
        2)
            prepareLangFlag=1
            shift
            ;;
        3)
            extractFeaturesFlag=1
            shift
            ;;
        4)
            monoTrainFlag=1
            shift
            ;;
        5)
            triTrainFlag=1
            shift
            ;;
        6)
            ubmTrainFlag=1
            shift
            ;;
        7)
            sgmmTrainFlag=1
            shift
            ;;
        8)
            sgmmTestFlag=1
            shift
            ;;
        9)
            deltasTrainFlag=1
            shift
            ;;
        10)
            finalVarUpdateFlag=1
            shift
            ;;
        --no-covar-update)
            noCovarUpdateFlag=1
            shift
            ;;
        11)
          triTestFlag=1
          shift
          ;;
        20)
            force=1
            shift
            ;;
        *)
            echo "ERROR: Unexpected parameter $1!"
            echo "Usage: run_basic.sh <options>"
            echo "1 Prepare Data"
            echo "2 Prepare Language Model"
            echo "3 Extract MFCC"
            echo "4 Monophone train"
            echo "5 Triphone train and test"
            echo "6 UBM train"
            echo "7 SGMM train"
            echo "8 SGMM test"
            echo "9 Deltas train and test"
            echo "10 Final Variance Update"
            echo "20 Force"
            exit 1
    esac
done

rm -f exp
mkdir -p $TEMP_DIR/exp || exit 1
ln -s $TEMP_DIR/exp . || exit 1

if [ $prepareDataFlag == 1 ]; then

    sh -x local/au4_data_prep.sh --use_htk_features $use_htk_features \
    --au4-data-dir $au4 --wsj0-data-dir $wsj0 \
    --wsj-lmdir $wsj_lmdir --dest-data-dir $DATA_DIR --use-no-spkr true \
    --dir $data || exit 1;

    # Sometimes, we have seen WSJ distributions that do not have subdirectories 
    # like '11-13.1', but instead have 'doc', 'si_et_05', etc. directly under the 
    # wsj0 or wsj1 directories. In such cases, try the following:
    #
    # corpus=/exports/work/inf_hcrc_cstr_general/corpora/wsj
    # local/cstr_wsj_data_prep.sh $corpus
    #
    # $corpus must contain a 'wsj0' and a 'wsj1' subdirectory for this to work.

fi

if [ $prepareLangFlag == 1 ]; then
    
    sh -x local/au4_prepare_sphinxdict.sh --data-dir $data || exit 1;

    utils/prepare_lang.sh  --position-dependent-phones true \
        $data/local/dict "<SPOKEN_NOISE>" $data/local/lang $data/lang || exit 1;

    sh -x local/au4_format_data.sh --data-dir $data || exit 1;

fi

 # We suggest to run the next three commands in the background,
 # as they are not a precondition for the system building and
 # most of the tests: these commands build a dictionary
 # containing many of the OOVs in the WSJ LM training data,
 # and an LM trained directly on that data (i.e. not just
 # copying the arpa files from the disks from LDC).
 # Caution: the commands below will only work if $decode_cmd 
 # is setup to use qsub.  Else, just remove the --cmd option.
 # NOTE: If you have a setup corresponding to the cstr_wsj_data_prep.sh style,
 # use local/cstr_wsj_extend_dict.sh $corpus/wsj1/doc/ instead.
# (
#  local/wsj_extend_dict.sh $wsj1/13-32.1  && \
#  utils/prepare_lang.sh $data/local/dict_larger "<SPOKEN_NOISE>" $data/local/lang_larger $data/lang_bd && \
#  local/wsj_train_lms.sh && \
#  local/wsj_format_local_lms.sh && 
#   (  local/wsj_train_rnnlms.sh --cmd "$decode_cmd -l mem_free=10G" $data/local/rnnlm.h30.voc10k &
#       sleep 20; # wait till tools compiled.
#     local/wsj_train_rnnlms.sh --cmd "$decode_cmd -l mem_free=12G" \
#      --hidden 100 --nwords 20000 --class 350 --direct 1500 $data/local/rnnlm.h100.voc20k &
#     local/wsj_train_rnnlms.sh --cmd "$decode_cmd -l mem_free=14G" \
#      --hidden 200 --nwords 30000 --class 350 --direct 1500 $data/local/rnnlm.h200.voc30k &
#     local/wsj_train_rnnlms.sh --cmd "$decode_cmd -l mem_free=16G" \
#      --hidden 300 --nwords 40000 --class 400 --direct 2000 $data/local/rnnlm.h300.voc40k &
#   )
# ) &

if [ $extractFeaturesFlag == 1 ]; then

    # Now make MFCC features.
    # mfccdir should be some place with a largish disk where you
    # want to store MFCC features.
    mkdir -p $mfccdir

    for i in `find data -maxdepth 1 -name "train_*" -type d`; do
        x=${i##*/}
        if [ ! $use_htk_features ]; then
            steps/make_mfcc.sh --cmd "$train_cmd" --nj $NP_feat \
                --mfcc-config $mfcc_config \
                $data/$x exp/make_mfcc/$x $mfccdir || exit 1;
        else
            steps/copy_htk_mfcc.sh --cmd "$train_cmd" --nj $NP_feat \
                $data/$x exp/make_mfcc/$x $mfccdir || exit 1
        fi

        steps/compute_cmvn_stats.sh $data/$x exp/make_mfcc/$x $mfccdir || exit 1;
    done

    for i in `find data -maxdepth 1 -name "test_*" -type d`; do
        x=${i##*/}
        if [ ! $use_htk_features ]; then
            steps/make_mfcc.sh --cmd "$train_cmd" --nj $NP_feat \
                --mfcc-config $mfcc_config \
                $data/$x exp/make_mfcc/$x $mfccdir || exit 1;
        else
            steps/copy_htk_mfcc.sh --cmd "$train_cmd" --nj $NP_feat \
                $data/$x exp/make_mfcc/$x $mfccdir || exit 1
        fi
        steps/compute_cmvn_stats.sh $data/$x exp/make_mfcc/$x $mfccdir || exit 1;
    done

fi

if [ $monoTrainFlag == 1 ]; then

    #utils/subset_data_dir.sh --first $data/train_si284 7138 $data/train_si84 || exit 1

    # Now make subset with the shortest 2k utterances from si-84.
    #utils/subset_data_dir.sh --shortest $data/train_si84 2000 $data/train_si84_2kshort || exit 1;

    # Now make subset with half of the data from si-84.
    utils/subset_data_dir.sh $data/train_${train_cond} 3500 $data/train_${train_cond}_half || exit 1;

    # Note: the --boost-silence option should probably be omitted by default
    # for normal setups.  It doesn't always help. [it's to discourage non-silence
    # models from modeling silence.]

    #steps/train_mono.sh --boost-silence 1.0 --nj $NP_train --cmd "$train_cmd" \
    #    $data/train_${train_cond}_half $data/lang $monoDir || exit 1
    
    steps/train_mono.sh --boost-silence 1.0 --nj $NP_train --cmd "$train_cmd" --feat_config $feat_config \
      --num-iters $num_iters --stage $stage \
        $data/train_${train_cond}_half $data/lang $monoDir || exit 1
    
    #utils/mkgraph.sh --mono $data/lang_test_$lm_suffix $monoDir $monoDir/graph_$lm_suffix


    #for i in `find data -maxdepth 1 -name "test_*" -type d`; do
    #    x=${i##*/}
    #    steps/decode.sh --nj $NP_decode --cmd "$decode_cmd" --feat_config $feat_config \
    #        $monoDir/graph_$lm_suffix $i $monoDir/decode_${lm_suffix}_${x} \
    #        || exit 1
    #done

fi

if [ $triTrainFlag == 1 ]; then

    if [ ! -f ${monoDir}_ali/final.mdl ] || [ $force == 1 ]; then
        steps/align_si.sh --boost-silence 1.0 --nj $NP_train --cmd "$train_cmd" --feat_config $feat_config \
            $data/train_${train_cond}_half $data/lang ${monoDir} ${monoDir}_ali || exit 1
    fi

    steps/train_deltas.sh --boost-silence 1.0 --cmd "$train_cmd" --feat_config $feat_config \
        $initGaussians $finalGaussians $data/train_$train_cond $data/lang ${monoDir}_ali $tri1Dir || exit 1

fi

if [ $triTestFlag == 1 ]; then
    
    # Align tri1 system with si84 data.
    steps/align_si.sh --nj $NP_train --cmd "$train_cmd" --feat_config $feat_config \
        $data/train_$train_cond $data/lang $tri1Dir ${tri1Dir}_ali || exit 1

    while [ ! -f $data/lang_test_$lm_suffix/tmp/LG.fst ] || \
        [ -z $data/lang_test_$lm_suffix/tmp/LG.fst ]; do
        sleep 20
    done
    sleep 30;
    # or the mono mkgraph.sh might be writing 
    # $data/lang_test_tgpr/tmp/LG.fst which will cause this to fail.

    utils/mkgraph.sh $data/lang_test_$lm_suffix $tri1Dir $tri1Dir/graph_$lm_suffix || exit 1

    for i in `find data -maxdepth 1 -name "test_*" -type d`; do
        x=${i##*/}
        steps/decode.sh --nj $NP_decode --cmd "$decode_cmd" --feat_config $feat_config \
            $tri1Dir/graph_$lm_suffix $i $tri1Dir/decode_${lm_suffix}_${x} \
            || exit 1
    done
    
    rm $decode_${lm_suffix}_${x}/gselect.*.gz 2> /dev/null
    rm $decode_${lm_suffix}_${x}/lat.*.gz 2> /dev/null
    
    mkdir -p results/tri1
    . cmd.sh
    #print_results $tri1Dir > results/tri1/${tri1Dir##*/}

    # test various modes of LM rescoring (4 is the default one).
    # This is just confirming they're equivalent.
    #for mode in 1 2 3 4; do
    #    steps/lmrescore.sh --mode $mode --cmd "$decode_cmd" $data/lang_test_$lm_suffix \
        #        $data/test_dev93 exp/tri1/decode_tgpr_dev93 exp/tri1/decode_tgpr_dev93_tg$mode  || exit 1
    #done

    # demonstrate how to get lattices that are "word-aligned" (arcs coincide with
    # words, with boundaries in the right place).
    #sil_label=`grep '!SIL' $data/lang_test_tgpr/words.txt | awk '{print $2}'`
    #steps/word_align_lattices.sh --cmd "$train_cmd" --silence-label $sil_label \
        #  $data/lang_test_tgpr exp/tri1/decode_tgpr_dev93 exp/tri1/decode_tgpr_dev93_aligned || exit 1

fi

if [ $deltasTrainFlag == 1 ]; then
    if [ ! -f ${tri1Dir}_ali/final.mdl ] || [ $force == 1 ]; then
        steps/align_si.sh --nj $NP_train --cmd "$train_cmd" --feat_config $feat_config \
            $data/train_$train_cond $data/lang $tri1Dir ${tri1Dir}_ali || exit 1
    fi

# Train tri2a, which is deltas + delta-deltas, on si84 data.
    steps/train_deltas.sh --cmd "$train_cmd" --feat_config $feat_config \
        $initGaussiansDeltas $finalGaussiansDeltas \
        $data/train_$train_cond $data/lang ${tri1Dir}_ali \
        $triDeltasDir || exit 1;
        
    steps/align_si.sh --nj $NP_train --cmd "$train_cmd" --feat_config $feat_config \
        $data/train_$train_cond $data/lang $triDeltasDir ${triDeltasDir}_ali || exit 1

    utils/mkgraph.sh $data/lang_test_${lm_suffix} $triDeltasDir $triDeltasDir/graph_$lm_suffix || exit 1

    for i in `find data -maxdepth 1 -name "test_*" -type d`; do
        x=${i##*/}
        steps/decode.sh --nj $NP_decode --cmd "$decode_cmd" \
            --feat_config $feat_config \
            $triDeltasDir/graph_$lm_suffix $i $triDeltasDir/decode_${lm_suffix}_${x} \
            || exit 1
    done
fi

sgmm_opts=
[ $ubmTrainFlag == 1 ] && sgmm_opts="$sgmm_opts 1"
[ $sgmmTrainFlag == 1 ] && sgmm_opts="$sgmm_opts 2"
[ $sgmmTestFlag == 1 ] && sgmm_opts="$sgmm_opts 3"
[ $finalVarUpdateFlag == 1 ] && sgmm_opts="$sgmm_opts 5"
[ $noCovarUpdateFlag == 1 ] && sgmm_opts="--no-covar-update $sgmm_opts"

local/run_sgmm_basic.sh $sgmm_opts || exit 1

echo "Aurora 4 experiment done!"
exit 0

#steps/train_lda_mllt.sh --cmd "$train_cmd" \
#   --splice-opts "--left-context=3 --right-context=3" \
#   2500 15000 $data/train_si84 $data/lang exp/tri1_ali_si84 exp/tri2b || exit 1;
#
#utils/mkgraph.sh $data/lang_test_tgpr exp/tri2b exp/tri2b/graph_tgpr || exit 1;
#steps/decode.sh --nj 10 --cmd "$decode_cmd" \
#  exp/tri2b/graph_tgpr $data/test_dev93 exp/tri2b/decode_tgpr_dev93 || exit 1;
#steps/decode.sh --nj 8 --cmd "$decode_cmd" \
#  exp/tri2b/graph_tgpr $data/test_eval92 exp/tri2b/decode_tgpr_eval92 || exit 1;
#
## Now, with dev93, compare lattice rescoring with biglm decoding,
## going from tgpr to tg.  Note: results are not the same, even though they should
## be, and I believe this is due to the beams not being wide enough.  The pruning
## seems to be a bit too narrow in the current scripts (got at least 0.7% absolute
## improvement from loosening beams from their current values).
#
#steps/decode_biglm.sh --nj 10 --cmd "$decode_cmd" \
#  exp/tri2b/graph_tgpr $data/lang_test_{tgpr,tg}/G.fst \
#  $data/test_dev93 exp/tri2b/decode_tgpr_dev93_tg_biglm
#
## baseline via LM rescoring of lattices.
#steps/lmrescore.sh --cmd "$decode_cmd" $data/lang_test_tgpr/ $data/lang_test_tg/ \
#  $data/test_dev93 exp/tri2b/decode_tgpr_dev93 exp/tri2b/decode_tgpr_dev93_tg || exit 1;
#
## Trying Minimum Bayes Risk decoding (like Confusion Network decoding):
#mkdir exp/tri2b/decode_tgpr_dev93_tg_mbr 
#cp exp/tri2b/decode_tgpr_dev93_tg/lat.*.gz exp/tri2b/decode_tgpr_dev93_tg_mbr 
#local/score_mbr.sh --cmd "$decode_cmd" \
# $data/test_dev93/ $data/lang_test_tgpr/ exp/tri2b/decode_tgpr_dev93_tg_mbr
#
#steps/decode_fromlats.sh --cmd "$decode_cmd" \
#  $data/test_dev93 $data/lang_test_tgpr exp/tri2b/decode_tgpr_dev93 \
#  exp/tri2a/decode_tgpr_dev93_fromlats || exit 1;
#
#
#
## Align tri2b system with si84 data.
#steps/align_si.sh  --nj 10 --cmd "$train_cmd" \
#  --use-graphs true $data/train_si84 $data/lang exp/tri2b exp/tri2b_ali_si84  || exit 1;
#
#
#local/run_mmi_tri2b.sh
#
#
## From 2b system, train 3b which is LDA + MLLT + SAT.
#steps/train_sat.sh --cmd "$train_cmd" \
#  2500 15000 $data/train_si84 $data/lang exp/tri2b_ali_si84 exp/tri3b || exit 1;
#utils/mkgraph.sh $data/lang_test_tgpr exp/tri3b exp/tri3b/graph_tgpr || exit 1;
#steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
#  exp/tri3b/graph_tgpr $data/test_dev93 exp/tri3b/decode_tgpr_dev93 || exit 1;
#steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
#  exp/tri3b/graph_tgpr $data/test_eval92 exp/tri3b/decode_tgpr_eval92 || exit 1;
#
## At this point you could run the command below; this gets
## results that demonstrate the basis-fMLLR adaptation (adaptation
## on small amounts of adaptation data).
#local/run_basis_fmllr.sh
#
#steps/lmrescore.sh --cmd "$decode_cmd" $data/lang_test_tgpr $data/lang_test_tg \
#  $data/test_dev93 exp/tri3b/decode_tgpr_dev93 exp/tri3b/decode_tgpr_dev93_tg || exit 1;
#steps/lmrescore.sh --cmd "$decode_cmd" $data/lang_test_tgpr $data/lang_test_tg \
#  $data/test_eval92 exp/tri3b/decode_tgpr_eval92 exp/tri3b/decode_tgpr_eval92_tg || exit 1;
#
#
## Trying the larger dictionary ("big-dict"/bd) + locally produced LM.
#utils/mkgraph.sh $data/lang_test_bd_tgpr exp/tri3b exp/tri3b/graph_bd_tgpr || exit 1;
#
#steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 8 \
#  exp/tri3b/graph_bd_tgpr $data/test_eval92 exp/tri3b/decode_bd_tgpr_eval92 || exit 1;
#steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 10 \
#  exp/tri3b/graph_bd_tgpr $data/test_dev93 exp/tri3b/decode_bd_tgpr_dev93 || exit 1;
#
#steps/lmrescore.sh --cmd "$decode_cmd" $data/lang_test_bd_tgpr $data/lang_test_bd_fg \
#  $data/test_eval92 exp/tri3b/decode_bd_tgpr_eval92 exp/tri3b/decode_bd_tgpr_eval92_fg \
#   || exit 1;
#steps/lmrescore.sh --cmd "$decode_cmd" $data/lang_test_bd_tgpr $data/lang_test_bd_tg \
#  $data/test_eval92 exp/tri3b/decode_bd_tgpr_eval92 exp/tri3b/decode_bd_tgpr_eval92_tg \
#  || exit 1;
#
#local/run_rnnlms_tri3b.sh
#
## The following two steps, which are a kind of side-branch, try mixing up
#( # from the 3b system.  This is to demonstrate that script.
# steps/mixup.sh --cmd "$train_cmd" \
#   20000 $data/train_si84 $data/lang exp/tri3b exp/tri3b_20k || exit 1;
# steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 10 \
#   exp/tri3b/graph_tgpr $data/test_dev93 exp/tri3b_20k/decode_tgpr_dev93  || exit 1;
#)
#
#
## From 3b system, align all si284 data.
#steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
#  $data/train_si284 $data/lang exp/tri3b exp/tri3b_ali_si284 || exit 1;
#
#
## From 3b system, train another SAT system (tri4a) with all the si284 data.
#
#steps/train_sat.sh  --cmd "$train_cmd" \
#  4200 40000 $data/train_si284 $data/lang exp/tri3b_ali_si284 exp/tri4a || exit 1;
#(
# utils/mkgraph.sh $data/lang_test_tgpr exp/tri4a exp/tri4a/graph_tgpr || exit 1;
# steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
#   exp/tri4a/graph_tgpr $data/test_dev93 exp/tri4a/decode_tgpr_dev93 || exit 1;
# steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
#   exp/tri4a/graph_tgpr $data/test_eval92 exp/tri4a/decode_tgpr_eval92 || exit 1;
#) &
#steps/train_quick.sh --cmd "$train_cmd" \
#   4200 40000 $data/train_si284 $data/lang exp/tri3b_ali_si284 exp/tri4b || exit 1;
#
#(
# utils/mkgraph.sh $data/lang_test_tgpr exp/tri4b exp/tri4b/graph_tgpr || exit 1;
# steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
#   exp/tri4b/graph_tgpr $data/test_dev93 exp/tri4b/decode_tgpr_dev93 || exit 1;
# steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
#  exp/tri4b/graph_tgpr $data/test_eval92 exp/tri4b/decode_tgpr_eval92 || exit 1;
#
# utils/mkgraph.sh $data/lang_test_bd_tgpr exp/tri4b exp/tri4b/graph_bd_tgpr || exit 1;
# steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
#   exp/tri4b/graph_bd_tgpr $data/test_dev93 exp/tri4b/decode_bd_tgpr_dev93 || exit 1;
# steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
#  exp/tri4b/graph_bd_tgpr $data/test_eval92 exp/tri4b/decode_bd_tgpr_eval92 || exit 1;
#) &
#
#
## Train and test MMI, and boosted MMI, on tri4b (LDA+MLLT+SAT on
## all the data).  Use 30 jobs.
#steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
#  $data/train_si284 $data/lang exp/tri4b exp/tri4b_ali_si284 || exit 1;
#
#local/run_mmi_tri4b.sh

## Segregated some SGMM builds into a separate file.
#local/run_sgmm.sh

# You probably want to run the sgmm2 recipe as it's generally a bit better:
#local/run_sgmm2.sh

# You probably wany to run the hybrid recipe as it is complementary:
#local/run_hybrid.sh


# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done


# KWS setup. We leave it commented out by default
# $duration is the length of the search collection, in seconds
#duration=`feat-to-len scp:$data/test_eval92/feats.scp  ark,t:- | awk '{x+=$2} END{print x/100;}'`
#local/generate_example_kws.sh $data/test_eval92/ $data/kws/
#local/kws_data_prep.sh $data/lang_test_bd_tgpr/ $data/test_eval92/ $data/kws/
#
#steps/make_index.sh --cmd "$decode_cmd" --acwt 0.1 \
#  $data/kws/ $data/lang_test_bd_tgpr/ \
#  exp/tri4b/decode_bd_tgpr_eval92/ \
#  exp/tri4b/decode_bd_tgpr_eval92/kws
#
#steps/search_index.sh --cmd "$decode_cmd" \
#  $data/kws \
#  exp/tri4b/decode_bd_tgpr_eval92/kws
#
# If you want to provide the start time for each utterance, you can use the --segments
# option. In WSJ each file is an utterance, so we don't have to set the start time.
#cat exp/tri4b/decode_bd_tgpr_eval92/kws/result.* | \
#  utils/write_kwslist.pl --flen=0.01 --duration=$duration \
#  --normalize=true --map-utter=$data/kws/utter_map \
#  - exp/tri4b/decode_bd_tgpr_eval92/kws/kwslist.xml


