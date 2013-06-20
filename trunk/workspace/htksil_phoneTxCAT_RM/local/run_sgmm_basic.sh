#!/bin/bash

#. cmd.sh

train_cmd="queue.pl"
decode_cmd="queue.pl"

initSubStates=0
finalSubStates=0

cdhmmName=tri1.vimal.try1.1800.9000
ubmName=ubm.vimal.try1
sgmmExptName=sgmm.vimal.try1.${initSubStates}.${finalSubStates}

ubmDir=exp/ubm/$ubmName
cdhmmDir=exp/tri1/${cdhmmName}
sgmmDir=exp/sgmm/$sgmmExptName

decode_config=conf/decode.config

NP_decode=20
NP_train=50

## SGMM on top of Delta+Delta-Deltas features.
if [ ! -e $ubmDir/final.mdl ]; then
  steps/train_ubm.sh --silence-weight 0.5 --cmd "$train_cmd" 400 data/train data/lang ${cdhmmDir}_ali $ubmDir || exit 1;
fi
steps/train_sgmm.sh --cmd "$train_cmd" $initSubStates $finalSubStates data/train data/lang ${cdhmmDir}_ali $ubmDir/final.ubm $sgmmDir || exit 1;

utils/mkgraph.sh data/lang $sgmmDir $sgmmDir/graph || exit 1;

steps/decode_sgmm.sh --config $decode_config --nj $NP_decode --cmd "$decode_cmd" \
  $sgmmDir/graph data/test $sgmmDir/decode || exit 1;

steps/decode_sgmm.sh --config $decode_config --nj 10 --cmd "$decode_cmd" \
  $sgmmDir/graph data/test_feb89 $sgmmDir/decode_feb89 || exit 1;

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

