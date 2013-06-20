#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# SGMM training, with speaker vectors.  This script would normally be called on
# top of fMLLR features obtained from a conventional system, but it also works
# on top of any type of speaker-independent features (based on
# deltas+delta-deltas or LDA+MLLT).  For more info on SGMMs, see the paper "The
# subspace Gaussian mixture model--A structured model for speech recognition".
# (Computer Speech and Language, 2011).

# Begin configuration section.
nj=4
cmd=queue.pl
update_cmd=queue.pl
stage=-6
context_opts= # e.g. set it to "--context-width=5 --central-position=2"  for a
# quinphone system.
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
num_iters=25   # Total number of iterations
realign_iters="5 10 15"; # Iters to realign on. 
#realign_iters=
max_iter_inc=15
rand_prune=0.1 # Randomized-pruning parameter for posteriors, to speed up training.
power=0.25 # Exponent for number of gaussians according to occurrence counts
beam=8
retry_beam=40
cluster_thresh=-1  # for build-tree control final bottom-up clustering of leaves
feat_config=conf/feat.config
gselect=50
full_gselect=15
use_state_dep_map=false
use_weight_projection=true
use_full_covar=false
num_transform_classes=1
num_cluster_weight_classes=1
train_scheme=1
cluster_transforms_iters=1
use_sequential_transform_update=false
num_threads=8
use_block_diagonal_transform=false
use_diagonal_transform=false
use_block_diagonal_transform2=false
use_diagonal_transform2=false
use_block_diagonal_transform3=false
use_diagonal_transform3=false
roots_file=
use_class_dep_steps=false
recluster_iters=
reinitializeA=false
use_sequential_multiple_txclass_update=false

# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 7 ]; then
  echo "Usage: steps/train_phoneTxCAT.sh <num-leaves> <num-substates> <data> <lang> <ali-dir> <ubm> <exp-dir>"
  echo " e.g.: steps/train_phoneTxCAT.sh 3500 10000 data/train_si84 data/lang \\"
  echo "                      exp/tri3b_ali_si84 exp/ubm4a/final.ubm exp/phoneTxCAT4a"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --silence-weight <sil-weight>                    # weight for silence (e.g. 0.5 or 0.0)"
  echo "  --num-iters <#iters>                             # Number of iterations of E-M"
  exit 1;
fi


num_leaves=$1
totsubstates=$2
data=$3
lang=$4
alidir=$5
ubm=$6
dir=$7

# Check some files.
for f in $data/feats.scp $lang/L.fst $alidir/ali.1.gz $alidir/final.mdl $ubm; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done


# Set some variables.
oov=`cat $lang/oov.int`
silphonelist=`cat $lang/phones/silence.csl`
ciphonelist=`cat $lang/phones/context_indep.csl` || exit 1;
numsubstates=$num_leaves # Initial #-substates.
incsubstates=0
feat_dim=`gmm-info $alidir/final.mdl 2>/dev/null | awk '/feature dimension/{print $NF}'` || exit 1;
[ $feat_dim -eq $feat_dim ] || exit 1; # make sure it's numeric.
[ -z $phn_dim ] && phn_dim=$[$feat_dim+1]
[ -z $spk_dim ] && spk_dim=$feat_dim
nj=`cat $alidir/num_jobs` || exit 1;

mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split$nj;
splice_opts=`cat $alidir/splice_opts 2>/dev/null`
cp $alidir/splice_opts $dir 2>/dev/null
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

gselect_opt="--gselect=ark,s,cs:gunzip -c $dir/gselect.JOB.gz|"

## Set up features.
if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) 
  
  if [[ -f $feat_config ]]; then

      feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas --config=$feat_config ark:- ark:- |"
  
  else
  
      feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |"
  
  fi
  ;;
  
  lda) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
    cp $alidir/final.mat $dir    
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac
if [ -f $alidir/trans.1 ]; then
  echo "$0: using transforms from $alidir"
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$alidir/trans.JOB ark:- ark:- |"
fi
##


if [ $stage -le -6 ]; then
  echo "$0: accumulating tree stats"
  $cmd JOB=1:$nj $dir/log/acc_tree.JOB.log \
    acc-tree-stats  --ci-phones=$ciphonelist $alidir/final.mdl "$feats" \
    "ark:gunzip -c $alidir/ali.JOB.gz|" $dir/JOB.treeacc || exit 1;
  [ "`ls $dir/*.treeacc | wc -w`" -ne "$nj" ] && echo "$0: Wrong #tree-stats" && exit 1;
  sum-tree-stats $dir/treeacc $dir/*.treeacc 2>$dir/log/sum_tree_acc.log || exit 1;
  rm $dir/*.treeacc
fi

if [ $stage -le -5 ]; then
  echo "$0: Getting questions for tree clustering."
  # preparing questions, roots file...
  cluster-phones $dir/treeacc $lang/phones/sets.int $dir/questions.int 2> $dir/log/questions.log || exit 1;
  cat $lang/phones/extra_questions.int >> $dir/questions.int
  compile-questions $lang/topo $dir/questions.int $dir/questions.qst 2>$dir/log/compile_questions.log || exit 1;

  echo "$0: Building the tree"
  $cmd $dir/log/build_tree.log \
    build-tree --verbose=1 --max-leaves=$num_leaves \
    --cluster-thresh=$cluster_thresh $dir/treeacc $lang/phones/roots.int \
    $dir/questions.qst $lang/topo $dir/tree || exit 1;
fi

if [ $stage -le -4 ]; then
  echo "$0: Initializing the model"  
  # Note: if phn_dim > feat_dim+1 or spk_dim > feat_dim, these dims
  # will be truncated on initialization.
  $cmd $dir/log/init_phoneTxCAT.log \
    phoneTxCAT-init --use-state-dep-map=$use_state_dep_map --use-weight-projection=$use_weight_projection --binary=false \
    --num-transform-classes=$num_transform_classes --use-full-covar=$use_full_covar --num-cluster-weight-classes=$num_cluster_weight_classes \
    --roots-file=$roots_file \
    $lang/topo \
    $dir/tree $dir/treeacc $ubm $dir/0.mdl || exit 1;
fi

if [ $stage -le -3 ]; then
  echo "$0: doing Gaussian selection"
  $cmd JOB=1:$nj $dir/log/gselect.JOB.log \
    phoneTxCAT-gselect --diag-gmm-nbest=$gselect --full-gmm-nbest=$full_gselect $dir/0.mdl "$feats" \
    "ark,t:|gzip -c >$dir/gselect.JOB.gz" || exit 1;
fi

if [ $stage -le -2 ]; then
  echo "$0: compiling training graphs"
  text="ark:sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $sdata/JOB/text|"
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
    compile-train-graphs $dir/tree $dir/0.mdl  $lang/L.fst  \
    "$text" "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
fi

if [ $stage -le -1 ]; then
  echo "$0: Converting alignments" 
  $cmd JOB=1:$nj $dir/log/convert_ali.JOB.log \
    convert-ali $alidir/final.mdl $dir/0.mdl $dir/tree "ark:gunzip -c $alidir/ali.JOB.gz|" \
    "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
fi

echo "Preparation complete!"

x=0
while [ $x -lt $num_iters ]; do
   echo "$0: training pass $x ... "
   if echo $realign_iters | grep -w $x >/dev/null && [ $stage -le $x ]; then
     echo "$0: re-aligning data"
     $cmd JOB=1:$nj $dir/log/align.$x.JOB.log  \
       phoneTxCAT-align-compiled $scale_opts "$gselect_opt" \
       --utt2spk=ark:$sdata/JOB/utt2spk --beam=$beam --retry-beam=$retry_beam \
       $dir/$x.mdl "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" \
       "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
   fi

#   if [ $[$x%5] -eq 0 ]; then
#       flags=Aw
#    elif [ $[$x%5] -eq 1 ]; then
#        flags=uS
#    elif [ $[$x%5] -eq 2 ]; then
#        flags=Aw
#    elif [ $[$x%5] -eq 3 ]; then
#        flags=uS
#    else
#        flags=v
#    fi
   if echo $recluster_iters | grep -w $x >/dev/null && [ $stage -le $x ]; then
     echo "$0: re-cluster gaussian"
     cp $dir/$x.mdl $dir/$x.mdl.tmp || exit 1;
     $cmd $dir/log/recluster.$x.log  \
     phoneTxCAT-init --copy-from-phoneTxCAT=true --recluster-gaussians=true \
     --reinitializeA=$reinitializeA --binary=false \
     $lang/topo \
     $dir/tree $dir/treeacc $dir/$x.mdl.tmp $dir/$x.mdl || exit 1;
   fi

    case $train_scheme in
        1)
           if [ $x -eq 0 ]; then
               flags=A 
           else
               if [ $[$x%2] -eq 1 ]; then
                   flags=AvwuS
               else
                   flags=A
               fi
           fi
           ;;
        2)
           if [ $x -eq 0 ]; then
               flags=A 
           else
               if [ $[$x%2] -eq 1 ]; then
                   flags=vwuS
               else
                   flags=A
               fi
           fi
           ;;
        3)
           if [ $x -eq 0 ]; then
               flags=A 
           else
               if [ $[$x%3] -eq 2 ]; then
                   flags=AvwuS
               else
                   flags=A
               fi
           fi
           ;;
        4)
           if [ $x -eq 0 ]; then
               flags=A 
           else
               if [ $[$x%3] -eq 2 ]; then
                   flags=vwuS
               else
                   flags=A
               fi
           fi
           ;;
        5)
           if [ $x -eq 0 ]; then
               flags=A 
           else
               flags=AvwuS
           fi
           ;;
        *)
           if [ $x -eq 0 ]; then
               flags=A 
           else
               if [ $[$x%2] -eq 1 ]; then
                   flags=AvwuS
               else
                   flags=A
               fi
           fi
    esac

    flags="${flags}t"
    if echo $recluster_iters | grep -w $x >/dev/null && [ $stage -le $x ]; then
      flags=At
    fi

    if [ $stage -le $x ]; then
      $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      phoneTxCAT-acc-stats --utt2spk=ark:$sdata/JOB/utt2spk \
      --update-flags=$flags "$gselect_opt" --rand-prune=$rand_prune \
      $dir/$x.mdl "$feats" "ark,s,cs:gunzip -c $dir/ali.JOB.gz | ali-to-post ark:- ark:-|" \
      $dir/$x.JOB.acc || exit 1;
    fi


   if [ $stage -le $x ]; then
     "$update_cmd" $dir/log/sum_accs.$x.log \
        phoneTxCAT-sum-accs --binary=false $dir/$x.accs $dir/$x.*.acc || exit 1
     rm $dir/$x.*.acc 2>/dev/null
     if [ "$update_cmd" != "run.pl" ]; then
       queue_opts="-l nodes=1:ppn=$num_threads"
     fi
     "$update_cmd" $queue_opts $dir/log/update.$x.log \
       phoneTxCAT-est --num-threads=$num_threads --binary=false --update-flags=$flags \
         --cluster-transforms-iters=$cluster_transforms_iters --use-sequential-transform-update=$use_sequential_transform_update \
         --use-block-diagonal-transform=$use_block_diagonal_transform --use-diagonal-transform=$use_diagonal_transform \
         --use-block-diagonal-transform2=$use_block_diagonal_transform2 --use-diagonal-transform2=$use_diagonal_transform2 \
         --use-block-diagonal-transform3=$use_block_diagonal_transform3 --use-diagonal-transform3=$use_diagonal_transform3 \
         --use-class-dep-steps=$use_class_dep_steps --use-sequential-multiple-txclass-update=$use_sequential_multiple_txclass_update \
         --write-occs=$dir/$[$x+1].occs $dir/$x.mdl $dir/$x.accs \
       $dir/$[$x+1].mdl || exit 1;
     #rm $dir/$x.accs 2> /dev/null
   fi
   
   x=$[$x+1];
done

rm $dir/final.mdl $dir/final.occs 2>/dev/null
ln -s $x.mdl $dir/final.mdl
ln -s $x.occs $dir/final.occs

utils/summarize_warnings.pl $dir/log

echo Done
