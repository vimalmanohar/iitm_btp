#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# This script does decoding with an PhoneTxCAT system

# Begin configuration section.
stage=1
alignment_model=
nj=4 # number of decoding jobs.
acwt=0.1  # Just a default value, used for adaptation and beam-pruning..
cmd=run.pl
beam=15.0
gselect=50  # Number of Gaussian-selection indices for PhoneTxCATs.  [Note:
            # the first_pass_gselect variable is used for the 1st pass of
            # decoding and can be tighter.
full_gselect=15
first_pass_gselect=3 # Use a smaller number of Gaussian-selection indices in 
            # the 1st pass of decoding (lattice generation).
max_active=7000
lat_beam=8.0 # Beam we use in lattice generation.
vecs_beam=4.0 # Beam we use to prune lattices while getting posteriors for 
    # speaker-vector computation.  Can be quite tight (actually we could
    # probably just do best-path.
feat_config=conf/feat.config

# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: steps/decode_phoneTxCAT.sh [options] <graph-dir> <data-dir> <decode-dir>"
  echo " e.g.: steps/decode_phoneTxCAT.sh \\"
  echo "      exp/phoneTxCAT3a/graph_tgpr data/test_dev93 exp/phoneTxCAT3a/decode_dev93_tgpr"
  echo "main options (for others, see top of script file)"
  echo "  --alignment-model <ali-mdl>              # Model for the first-pass decoding."
  echo "  --config <config-file>                   # config containing options"
  echo "  --nj <nj>                                # number of parallel jobs"
  echo "  --cmd <cmd>                              # Command to run in parallel with"
  echo "  --beam <beam>                            # Decoding beam; default 13.0"
  exit 1;
fi

graphdir=$1
data=$2
dir=$3
srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.

for f in $graphdir/HCLG.fst $data/feats.scp $srcdir/final.mdl; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

sdata=$data/split$nj;
silphonelist=`cat $graphdir/phones/silence.csl` || exit 1
splice_opts=`cat $srcdir/splice_opts 2>/dev/null`
gselect_opt="--gselect=ark:gunzip -c $dir/gselect.JOB.gz|"
gselect_opt_1stpass="$gselect_opt copy-gselect --n=$first_pass_gselect ark:- ark:- |"

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs


## Set up features.
if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) 

  if [[ -f $feat_config ]]; then

      feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas --config=$feat_config ark:- ark:- |"
  
  else
  
      feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |"
  
  fi
  ;;
  lda) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac
##

## Save Gaussian-selection info to disk.
# Note: we can use final.mdl regardless of whether there is an alignment model--
# they use the same UBM.
if [ $stage -le 1 ]; then
  $cmd JOB=1:$nj $dir/log/gselect.JOB.log \
    phoneTxCAT-gselect --diag-gmm-nbest=$gselect --full-gmm-nbest=$full_gselect $srcdir/final.mdl \
    "$feats" "ark:|gzip -c >$dir/gselect.JOB.gz" || exit 1;
fi

## Work out name of alignment model. ##
if [ -z "$alignment_model" ]; then
  if [ -f "$srcdir/final.alimdl" ]; then alignment_model=$srcdir/final.alimdl;
  else alignment_model=$srcdir/final.mdl; fi
fi
[ ! -f "$alignment_model" ] && echo "$0: no alignment model $alignment_model " && exit 1;

# Generate state-level lattice which we can rescore.  This is done with the 
# alignment model and no speaker-vectors.
if [ $stage -le 2 ]; then
  $cmd JOB=1:$nj $dir/log/decode_pass1.JOB.log \
    phoneTxCAT-latgen-faster --max-active=$max_active --beam=$beam --lattice-beam=$lat_beam \
    --acoustic-scale=$acwt --determinize-lattice=false --allow-partial=true \
    --word-symbol-table=$graphdir/words.txt "$gselect_opt_1stpass" $alignment_model \
    $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $dir/pre_lat.JOB.gz" || exit 1;
fi

if [ $stage -le 3 ]; then
    for n in `seq 1 $nj`; do
        mv $dir/pre_lat.${n}.gz $dir/lat.${n}.gz
    done
fi

# The output of this script is the files "lat.*.gz"-- we'll rescore this at 
# different acoustic scales to get the final output.

if [ $stage -le 4 ]; then
  [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
  echo "score best paths"
  local/score.sh --cmd "$cmd" --min_lmwt 2 --max_lmwt 14 $data $graphdir $dir
  echo "score confidence and timing with sclite"
  #local/score_sclite_conf.sh --cmd "$cmd" --language turkish $data $graphdir $dir
fi
echo "Decoding done."
exit 0

