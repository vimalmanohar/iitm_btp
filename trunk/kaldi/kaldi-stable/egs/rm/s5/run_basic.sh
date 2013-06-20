#!/bin/bash

# CAUTION: I changed e.g. 1.trans to trans.1 in the scripts.  If you ran it
# part-way through prior to this, to convert to the new naming
# convention, run:
# for x in `find . -name '*.trans'`; do mv $x `echo $x | perl -ane 's/(\d+)\.trans/trans.$1/;print;'`; done
# but be careful as this will not follow soft links.

. path.sh

train_cmd="queue.pl"
decode_cmd="queue.pl"
run_cmd="queue.pl"

# call the next line with the directory where the RM data is
# (the argument below is just an example).  This should contain
# subdirectories named as follows:
#    rm1_audio1  rm1_audio2	rm2_audio

#local/rm_data_prep.sh /mnt/matylda2/data/RM || exit 1;

PATH_BASE=`pwd`

DATA_DIR=/speech/vimal/scratch/RM_kaldi/data
TEMP_DIR=/speech/vimal/scratch/RM_kaldi

NP_features=20
NP_train=50
NP_decode=20

initGaussians=1800          # Number of tied states
finalGaussians=9000         # Total number of Gaussian mixtures

decode_config=conf/decode.config

prefix=try2
stage=-6

if (( $# == 0 )); then
    echo "1) prepareTrainFlag"
    echo "2) extractFeaturesFlag"
    echo "3) monoTrainFlag"
    echo "4) triTrainFlag"
    echo "5) triTestFlag"
    exit 0
fi

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

monoName=mono.vimal.$prefix
tri1Name=tri1.vimal.$prefix.$initGaussians.$finalGaussians
triDeltasName=tri2a.vimal.$prefix.$initGaussians.$finalGaussians

monoDir=exp/mono/$monoName
tri1Dir=exp/tri1/$tri1Name
triDeltasDir=exp/triDeltas/$triDeltasName

prepareTrainFlag=0
extractFeaturesFlag=0
monoTrainFlag=0
triTrainFlag=0
triTestFlag=0
triDeltasFlag=0
triDeltasTestFlag=0

while (( $# > 0 )); do
    case $1 in
        1)
            prepareTrainFlag=1
            shift
            ;;
        2)
            extractFeaturesFlag=1
            shift
            ;;
        3)
            monoTrainFlag=1
            shift
            ;;
        4)
            triTrainFlag=1
            shift
            ;;
        5)
            triTestFlag=1
            shift
            ;;
        6)
            triDeltasFlag=1
            shift
            ;;
        7)
            triDeltasTestFlag=1
            shift
            ;;
        *)
            echo "ERROR: Invalid option!"
            echo "1) prepareTrainFlag"
            echo "2) extractFeaturesFlag"
            echo "3) monoTrainFlag"
            echo "4) triTrainFlag"
            echo "5) triTestFlag"
            echo "6) triDeltasFlag"
            echo "7) triDeltasTestFlag"
            exit 1
        esac
done


rm -f exp
mkdir -p $TEMP_DIR/exp || exit 1
ln -s $TEMP_DIR/exp . || exit 1

#DATA_DIR=/export/corpora5/LDC/LDC93S3A/rm_comp

if [ $prepareTrainFlag == 1 ]; then

    bash -x local/rm_data_prep.sh $DATA_DIR || exit 1;

    bash -x utils/prepare_lang.sh --position-dependent-phones true data/local/dict '!SIL' data/local/lang data/lang || exit 1;

    bash -x local/rm_prepare_grammar.sh || exit 1;

# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
fi

if [ $extractFeaturesFlag == 1 ]; then
    featdir=$TEMP_DIR/features

    for x in test_mar87 test_oct87 test_feb89 test_oct89 test_feb91 test_sep92 train; do
        steps/make_mfcc.sh --nj $NP_features --cmd "queue.pl" data/$x exp/make_mfcc/$x $featdir  || exit 1;
      steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $featdir  || exit 1;
      #steps/make_plp.sh data/$x exp/make_plp/$x $featdir 4
    done

# Make a combined data dir where the data from all the test sets goes-- we do
# all our testing on this averaged set.  This is just less hassle.  We
# regenerate the CMVN stats as one of the speakers appears in two of the 
# test sets; otherwise tools complain as the archive has 2 entries.
    utils/combine_data.sh data/test data/test_{mar87,oct87,feb89,oct89,feb91,sep92}
    steps/compute_cmvn_stats.sh data/test exp/make_mfcc/test $featdir  

    utils/subset_data_dir.sh data/train 1000 data/train.1k  || exit 1;
fi

if [ $monoTrainFlag == 1 ]; then

    #steps/train_mono.sh --nj 4 --cmd "$train_cmd" data/train.1k data/lang $monoDir  || exit 1;
    sh -x steps/train_mono.sh --nj $NP_train --cmd "$train_cmd" --stage $stage data/train.1k data/lang $monoDir || exit 1;

    #show-transitions data/lang/phones.txt $triDeltasDir/final.mdl  $triDeltasDir/final.occs | perl -e 'while(<>) { if (m/ sil /) { $l = <>; $l =~ m/pdf = (\d+)/|| die "bad line $l";  $tot += $1; }} print "Total silence count $tot\n";'

    utils/mkgraph.sh --mono data/lang $monoDir $monoDir/graph || exit 1

    ##steps/decode.sh --config $decode_config --nj 20 --cmd "$decode_cmd" \
    ##  $monoDir/graph data/test $monoDir/decode
    #    steps/decode.sh --config $decode_config --nj $NP_decode --cmd "$decode_cmd" \
    #      $monoDir/graph data/test $monoDir/decode
    #

    # Get alignments from monophone system.
    #steps/align_si.sh --nj 8 --cmd "$train_cmd" \
    #  data/train data/lang $monoDir ${monoDir}_ali || exit 1;
    steps/align_si.sh --nj $NP_train --cmd "$train_cmd" \
    data/train data/lang $monoDir ${monoDir}_ali || exit 1;
fi

if [ $triTrainFlag == 1 ]; then
    # train tri1 [first triphone pass]
    steps/train_deltas.sh --cmd "$train_cmd" \
    $initGaussians $finalGaussians data/train data/lang ${monoDir}_ali $tri1Dir || exit 1;
    
    # align tri1
    #steps/align_si.sh --nj 8 --cmd "$train_cmd" \
    #  --use-graphs true data/train data/lang $tri1Dir ${tri1Dir}_ali || exit 1;
    steps/align_si.sh --nj $NP_train --cmd "$train_cmd" \
    --use-graphs true data/train data/lang $tri1Dir ${tri1Dir}_ali || exit 1;
fi

if [ $triTestFlag == 1 ]; then

    # decode tri1
    queue.pl mkgraph.log utils/mkgraph.sh data/lang $tri1Dir $tri1Dir/graph || exit 1;
    #steps/decode.sh --config $decode_config --nj 20 --cmd "$decode_cmd" \
    #  $tri1Dir/graph data/test $tri1Dir/decode
    steps/decode.sh --config $decode_config --nj $NP_decode --cmd "$decode_cmd" \
    $tri1Dir/graph data/test $tri1Dir/decode

    #draw-tree data/lang/phones.txt $tri1Dir/tree | dot -Tps -Gsize=8,10.5 | ps2pdf - tree.pdf

fi

if [ $triDeltasFlag == 1 ]; then
    # train tri2a [delta+delta-deltas]
    steps/train_deltas.sh --cmd "$train_cmd" $initGaussians $finalGaussians \
    data/train data/lang ${tri1Dir}_ali $triDeltasDir || exit 1;
    
    steps/align_si.sh --nj $NP_train --cmd "$train_cmd" \
    --use-graphs true data/train data/lang $triDeltasDir ${triDeltasDir}_ali || exit 1;
fi

if [ $triDeltasTestFlag == 1 ]; then
    ## decode tri2a
    utils/mkgraph.sh data/lang $triDeltasDir $triDeltasDir/graph
    steps/decode.sh --config $decode_config --nj $NP_decode --cmd "$decode_cmd" \
    $triDeltasDir/graph data/test $triDeltasDir/decode
fi

echo "CDHMM model training done!"
echo "run local/run_sgmm_basic.sh for SGMM training"

exit 0
exit 0


###########################################################################
# The code below this is not ready yet
###########################################################################
# train and decode tri2b [LDA+MLLT]
steps/train_lda_mllt.sh --cmd "$train_cmd" \
  --splice-opts "--left-context=3 --right-context=3" \
 1800 9000 data/train data/lang ${tri1Dir}_ali exp/tri2b || exit 1;
utils/mkgraph.sh data/lang exp/tri2b exp/tri2b/graph
#steps/decode.sh --config $decode_config --nj 20 --cmd "$decode_cmd" \
#   exp/tri2b/graph data/test exp/tri2b/decode
steps/decode.sh --config $decode_config --nj $NP --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b/decode

# Align all data with LDA+MLLT system (tri2b)
#steps/align_si.sh --nj 8 --cmd "$train_cmd" --use-graphs true \
#   data/train data/lang exp/tri2b exp/tri2b_ali || exit 1;
steps/align_si.sh --nj $NP --cmd "$train_cmd" --use-graphs true \
   data/train data/lang exp/tri2b exp/tri2b_ali || exit 1;

#  Do MMI on top of LDA+MLLT.
#steps/make_denlats.sh --nj 8 --cmd "$train_cmd" \
#  data/train data/lang exp/tri2b exp/tri2b_denlats || exit 1;
steps/make_denlats.sh --nj $NP --cmd "$train_cmd" \
  data/train data/lang exp/tri2b exp/tri2b_denlats || exit 1;
steps/train_mmi.sh data/train data/lang exp/tri2b_ali exp/tri2b_denlats exp/tri2b_mmi || exit 1;
#steps/decode.sh --config $decode_config --iter 4 --nj 20 --cmd "$decode_cmd" \
#   exp/tri2b/graph data/test exp/tri2b_mmi/decode_it4
steps/decode.sh --config $decode_config --iter 4 --nj $NP --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mmi/decode_it4
#steps/decode.sh --config $decode_config --iter 3 --nj 20 --cmd "$decode_cmd" \
#   exp/tri2b/graph data/test exp/tri2b_mmi/decode_it3
steps/decode.sh --config $decode_config --iter 3 --nj $NP --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mmi/decode_it3

# Do the same with boosting.
steps/train_mmi.sh --boost 0.05 data/train data/lang \
   exp/tri2b_ali exp/tri2b_denlats exp/tri2b_mmi_b0.05 || exit 1;
#steps/decode.sh --config $decode_config --iter 4 --nj 20 --cmd "$decode_cmd" \
#   exp/tri2b/graph data/test exp/tri2b_mmi_b0.05/decode_it4 || exit 1;
steps/decode.sh --config $decode_config --iter 4 --nj $NP --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mmi_b0.05/decode_it4 || exit 1;
#steps/decode.sh --config $decode_config --iter 3 --nj 20 --cmd "$decode_cmd" \
#   exp/tri2b/graph data/test exp/tri2b_mmi_b0.05/decode_it3 || exit 1;
steps/decode.sh --config $decode_config --iter 3 --nj $NP --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mmi_b0.05/decode_it3 || exit 1;

# Do MPE.
steps/train_mpe.sh data/train data/lang exp/tri2b_ali exp/tri2b_denlats exp/tri2b_mpe || exit 1;
#steps/decode.sh --config $decode_config --iter 4 --nj 20 --cmd "$decode_cmd" \
#   exp/tri2b/graph data/test exp/tri2b_mpe/decode_it4 || exit 1;
steps/decode.sh --config $decode_config --iter 4 --nj $NP --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mpe/decode_it4 || exit 1;
#steps/decode.sh --config $decode_config --iter 3 --nj 20 --cmd "$decode_cmd" \
#   exp/tri2b/graph data/test exp/tri2b_mpe/decode_it3 || exit 1;
steps/decode.sh --config $decode_config --iter 3 --nj $NP --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mpe/decode_it3 || exit 1;


## Do LDA+MLLT+SAT, and decode.
steps/train_sat.sh 1800 9000 data/train data/lang exp/tri2b_ali exp/tri3b || exit 1;
utils/mkgraph.sh data/lang exp/tri3b exp/tri3b/graph || exit 1;
#steps/decode_fmllr.sh --config $decode_config --nj 20 --cmd "$decode_cmd" \
#  exp/tri3b/graph data/test exp/tri3b/decode || exit 1;
steps/decode_fmllr.sh --config $decode_config --nj $NP --cmd "$decode_cmd" \
  exp/tri3b/graph data/test exp/tri3b/decode || exit 1;



# Align all data with LDA+MLLT+SAT system (tri3b)
#steps/align_fmllr.sh --nj 8 --cmd "$train_cmd" --use-graphs true \
#  data/train data/lang exp/tri3b exp/tri3b_ali || exit 1;
steps/align_fmllr.sh --nj $NP --cmd "$train_cmd" --use-graphs true \
  data/train data/lang exp/tri3b exp/tri3b_ali || exit 1;

## MMI on top of tri3b (i.e. LDA+MLLT+SAT+MMI)
#steps/make_denlats.sh --config $decode_config \
#   --nj 8 --cmd "$train_cmd" --transform-dir exp/tri3b_ali \
#  data/train data/lang exp/tri3b exp/tri3b_denlats || exit 1;
steps/make_denlats.sh --config $decode_config \
   --nj $NP --cmd "$train_cmd" --transform-dir exp/tri3b_ali \
  data/train data/lang exp/tri3b exp/tri3b_denlats || exit 1;
steps/train_mmi.sh data/train data/lang exp/tri3b_ali exp/tri3b_denlats exp/tri3b_mmi || exit 1;

#steps/decode_fmllr.sh --config $decode_config --nj 20 --cmd "$decode_cmd" \
#  --alignment-model exp/tri3b/final.alimdl --adapt-model exp/tri3b/final.mdl \
#   exp/tri3b/graph data/test exp/tri3b_mmi/decode || exit 1;
steps/decode_fmllr.sh --config $decode_config --nj $NP --cmd "$decode_cmd" \
  --alignment-model exp/tri3b/final.alimdl --adapt-model exp/tri3b/final.mdl \
   exp/tri3b/graph data/test exp/tri3b_mmi/decode || exit 1;

# Do a decoding that uses the exp/tri3b/decode directory to get transforms from.
#steps/decode.sh --config $decode_config --nj 20 --cmd "$decode_cmd" \
#  --transform-dir exp/tri3b/decode  exp/tri3b/graph data/test exp/tri3b_mmi/decode2 || exit 1;
steps/decode.sh --config $decode_config --nj $NP --cmd "$decode_cmd" \
  --transform-dir exp/tri3b/decode  exp/tri3b/graph data/test exp/tri3b_mmi/decode2 || exit 1;


#first, train UBM for fMMI experiments.
#steps/train_diag_ubm.sh --silence-weight 0.5 --nj 8 --cmd "$train_cmd" \
#  250 data/train data/lang exp/tri3b_ali exp/dubm3b
steps/train_diag_ubm.sh --silence-weight 0.5 --nj $NP --cmd "$train_cmd" \
  250 data/train data/lang exp/tri3b_ali exp/dubm3b

# Next, various fMMI+MMI configurations.
steps/train_mmi_fmmi.sh --learning-rate 0.0025 \
  --boost 0.1 --cmd "$train_cmd" data/train data/lang exp/tri3b_ali exp/dubm3b exp/tri3b_denlats \
  exp/tri3b_fmmi_b || exit 1;

for iter in 3 4 5 6 7 8; do
 #steps/decode_fmmi.sh --nj 20 --config $decode_config --cmd "$decode_cmd" --iter $iter \
 #  --transform-dir exp/tri3b/decode  exp/tri3b/graph data/test exp/tri3b_fmmi_b/decode_it$iter &
 steps/decode_fmmi.sh --nj $NP --config $decode_config --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri3b/decode  exp/tri3b/graph data/test exp/tri3b_fmmi_b/decode_it$iter &
done

steps/train_mmi_fmmi.sh --learning-rate 0.001 \
  --boost 0.1 --cmd "$train_cmd" data/train data/lang exp/tri3b_ali exp/dubm3b exp/tri3b_denlats \
  exp/tri3b_fmmi_c || exit 1;

for iter in 3 4 5 6 7 8; do
 #steps/decode_fmmi.sh --nj 20 --config $decode_config --cmd "$decode_cmd" --iter $iter \
 #  --transform-dir exp/tri3b/decode  exp/tri3b/graph data/test exp/tri3b_fmmi_c/decode_it$iter &
 steps/decode_fmmi.sh --nj $NP --config $decode_config --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri3b/decode  exp/tri3b/graph data/test exp/tri3b_fmmi_c/decode_it$iter &
done

# for indirect one, use twice the learning rate.
steps/train_mmi_fmmi_indirect.sh --learning-rate 0.01 --schedule "fmmi fmmi fmmi fmmi mmi mmi mmi mmi" \
  --boost 0.1 --cmd "$train_cmd" data/train data/lang exp/tri3b_ali exp/dubm3b exp/tri3b_denlats \
  exp/tri3b_fmmi_d || exit 1;

for iter in 3 4 5 6 7 8; do
 #steps/decode_fmmi.sh --nj 20 --config $decode_config --cmd "$decode_cmd" --iter $iter \
 #  --transform-dir exp/tri3b/decode  exp/tri3b/graph data/test exp/tri3b_fmmi_d/decode_it$iter &
 steps/decode_fmmi.sh --nj $NP --config $decode_config --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri3b/decode  exp/tri3b/graph data/test exp/tri3b_fmmi_d/decode_it$iter &
done

# You don't have to run all 3 of the below, e.g. you can just run the run_sgmm2x.sh
local/run_sgmm.sh
local/run_sgmm2.sh
local/run_sgmm2x.sh

