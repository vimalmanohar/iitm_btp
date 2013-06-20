#!/bin/bash

# Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
set -o pipefail

if [ $# -le 7 ]; then
    echo "Usage: au4_data_prep.sh [options...]"
    echo "Options:"
    echo "  --au4-data-dir  : Path to Aurora4 original data containing"
    echo "                    folders like train_clean etc."
    echo "  --wsj0-data-dir : Path to WSJ0-SI84 containing folders"
    echo "                    like 011 corresponding to different spkrs"
    echo "  --wsj-lmdir     : Path to WSJ LANG_MODL directory"
    echo "  --dest-data-dir : Path to store converted Au4 wav files"
    echo "  --use-no-spkr   : Neglect speaker information"
    echo "  --dir           : Output data dir [data]"
    exit 1;
fi

dir=`pwd`/data
mkdir -p $dir $lmdir $dir/local
local=`pwd`/local
utils=`pwd`/utils

au4_data_dir=
wsj0_data_dir=
wsj_lmdir=
dest_data_dir=
use_no_spkr=false
use_htk_features=false

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

DATA_DIR=$dest_data_dir
wsj0_dir=$wsj0_data_dir

lmdir=$dir/lang


if [ ! $use_htk_features ]; then
    mkdir -p $DATA_DIR || exit 1
    [ ! -d "$DATA_DIR" ] && exit 1
fi

[ ! -d "$au4_data_dir" ] && exit 1
[ ! -d "$wsj0_dir" ] && exit 1
[ ! -d "$wsj_lmdir" ] && exit 1

. ./path.sh || exit 1
# Needed for KALDI_ROOT
#export PATH=$PATH:$KALDI_ROOT/tools/irstlm/bin
#sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
#if [ ! -x $sph2pipe ]; then
#   echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
#   exit 1;
#fi

#cd $dir

# Make directory of links to the WSJ disks such as 11-13.1.  This relies on the command
# line arguments being absolute pathnames.

# Do some basic checks that we have what we expected.
numTrainDir=`find $au4_data_dir/. -name "train_*" -type d | wc | awk -F" " '{print $1}'`;
numTestDir=`find $au4_data_dir/. -name "test_*" -type d | wc | awk -F" " '{print $1}'`;

if [ $numTrainDir != 3 ] || [ $numTestDir != 14 ]; then
  echo "au4_data_prep.sh: Spot check of command line arguments failed"
  echo "Command line arguments must be absolute pathnames to Aurora4 directories"
  echo "with names like train_clean, test_clean_wv1 etc."
  exit 1;
fi

export LC_ALL=C

tmpdir=$dir/local/tmp
mkdir -p $tmpdir || exit 1

sox="sox -r 8000 -c 1 -s -w -x"

count=0
for x in `find $au4_data_dir/. -maxdepth 1 -name "train_*" -type d`; do
    i=${x##*/}
    trainDirList[$count]=$i
    mkdir -p $dir/$i || exit 1
    if [ ! $use_htk_features ]; then
        mkdir -p $DATA_DIR/$i
        find $au4_data_dir/$i -maxdepth 1 -type d \
            | sed 's;'$au4_data_dir/$i';'$DATA_DIR/$i';g' \
            | awk '{system("mkdir -p "$1)}' || exit 1
        find $au4_data_dir/$i -name "*.wv[1-2]" \
            | sed 's;'$au4_data_dir'/\(.*\);'$au4_data_dir'/\1 '$DATA_DIR'/\1;g' \
            | sed 's;\.wv[1-2]$;.wav;g' \
            > $tmpdir/${i}.raw2wav.scp || exit 1
        if [ $( find $DATA_DIR/$i -name "*.wav" | wc | awk '{print $1}' ) \
            != $( wc $tmpdir/${i}.raw2wav.scp | awk '{print $1}' ) ]; then
            cat $tmpdir/${i}.raw2wav.scp | \
                awk -F" " '{system("'"$sox"' -t raw "$1" -t wav "$2)}' \
                || exit 1 
        fi
        find $DATA_DIR/$i -name "*.wav" > $tmpdir/${i}.flist.tmp || exit 1
    else
        find $au4_data_dir/$i -name "*.mfc" > $tmpdir/${i}.flist.tmp || exit 1
    fi

    cat $tmpdir/${i}.flist.tmp | sort | uniq > $tmpdir/${i}.flist || exit 1
    rm $tmpdir/${i}.flist.tmp || exit 1
    count=$((count+1))
done || exit 1

count=0
for x in `find $au4_data_dir/. -maxdepth 1 -name "test_*" -type d`; do
    i=${x##*/}
    testDirList[$count]=$i
    mkdir -p $dir/$i || exit 1
    if [ ! $use_htk_features ]; then
        mkdir -p $DATA_DIR/$i
        find $au4_data_dir/$i -maxdepth 1 -type d \
            | sed 's;'$au4_data_dir/$i';'$DATA_DIR/$i';g' \
            | awk '{system("mkdir -p "$1)}' || exit 1
        find $au4_data_dir/$i -name "*.wv[1-2]" \
            | sed 's;'$au4_data_dir'/\(.*\);'$au4_data_dir'/\1 '$DATA_DIR'/\1;g' \
            | sed 's;.wv[1-2]$;.wav;g' \
            > $tmpdir/${i}.raw2wav.scp || exit 1
        if [ $( find $DATA_DIR/$i -name "*.wav" | wc | awk '{print $1}' ) \
            != $( wc $tmpdir/${i}.raw2wav.scp | awk '{print $1}' ) ]; then
            cat $tmpdir/${i}.raw2wav.scp | \
                awk -F" " '{system("'"$sox"' -t raw "$1" -t wav "$2)}' \
                || exit 1 
        fi
        find $DATA_DIR/$i -name "*.wav" > $tmpdir/${i}.flist.tmp || exit 1
    else
        find $au4_data_dir/$i -name "*.mfc" > $tmpdir/${i}.flist.tmp || exit 1
    fi
    
    cat $tmpdir/${i}.flist.tmp | sort | uniq > $tmpdir/${i}.flist || exit 1
    rm $tmpdir/${i}.flist.tmp || exit 1
    count=$((count+1))
done || exit 1

# Finding the transcript files:
for i in $wsj0_dir; do find -L $i -iname '*.dot'; done \
    > $tmpdir/dot_files.flist || exit 1

noiseword="<NOISE>";

if [ ! $use_htk_features ]; then
    scpFile=wav.scp
else
    scpFile=mfc.scp
fi

for i in ${trainDirList[@]}; do
    if [ !$use_htk_features]; then
        $local/flist2scp.pl $tmpdir/$i.flist | sort > $dir/$i/$scpFile || exit 1
    else
        for x in $(cat $tmpdir/$i.flist); do
            y=${x##*/}
            z=${y%.*}
            echo "$z $x"
        done | sort > $dir/$i/$scpFile || exit 1
    fi
    cat $dir/$i/$scpFile | awk '{print $1}' | $local/find_transcripts.pl $tmpdir/dot_files.flist > $tmpdir/$i.trans1 || exit 1
    cat $tmpdir/$i.trans1 | $local/normalize_transcript.pl $noiseword | sort > $dir/$i/text || exit 1;
    #   awk '{printf("%s '"$sox"' -t raw %s -t wav - |\n", $1, $2);}' < $dir/$i/raw.scp > $dir/$i/$scpFile || exit 1
done || exit 1

for i in ${testDirList[@]}; do
    if [ !$use_htk_features]; then
        $local/flist2scp.pl $tmpdir/$i.flist | sort > $dir/$i/$scpFile || exit 1
    else
        for x in $(cat $tmpdir/$i.flist); do
            y=${x##*/}
            z=${y%.*}
            echo "$z $x"
        done | sort > $dir/$i/$scpFile || exit 1
    fi
    cat $dir/$i/$scpFile | awk '{print $1}' | $local/find_transcripts.pl $tmpdir/dot_files.flist > $tmpdir/$i.trans1 || exit 1
    cat $tmpdir/$i.trans1 | $local/normalize_transcript.pl $noiseword | sort > $dir/$i/text || exit 1;
#   awk '{printf("%s '"$sox"' -t raw %s -t wav - |\n", $1, $2);}' < $dir/$i/raw.scp > $dir/$i/$scpFile || exit 1
done || exit 1

# Make the utt2spk and spk2utt files.
for i in ${trainDirList[@]}; do
    if [ $use_no_spkr ]; then
        cat $dir/$i/$scpFile | awk '{print $1" "$1}' > $dir/$i/utt2spk
    else 
        cat $dir/$i/$scpFile | awk '{print $1}' \
            | perl -ane 'chop; m:^...:; print "$_ $&\n";' \
            > $dir/$i/utt2spk || exit 1
    fi || exit 1

    cat $dir/$i/utt2spk | $utils/utt2spk_to_spk2utt.pl \
        > $dir/$i/spk2utt || exit 1;
done || exit 1

for i in ${testDirList[@]}; do
    if [ $use_no_spkr ]; then
        cat $dir/$i/$scpFile | awk '{print $1" "$1}' > $dir/$i/utt2spk
    else 
        cat $dir/$i/$scpFile | awk '{print $1}' \
            | perl -ane 'chop; m:^...:; print "$_ $&\n";' \
            > $dir/$i/utt2spk || exit 1
    fi || exit 1

    cat $dir/$i/utt2spk | $utils/utt2spk_to_spk2utt.pl \
        > $dir/$i/spk2utt || exit 1;
done || exit 1

#rm -f $dir/local/lm.arpa 2> /dev/null
#cp -f $2 $dir/local/lm.arpa

#in case we want to limit lm's on most frequent words, copy lm training word frequency list
#cp links/13-32.1/wsj1/doc/lng_modl/vocab/wfl_64.lst $lmdir
#chmod u+w $lmdir/*.lst # had weird permissions on source.

# The 20K vocab, open-vocabulary language model (i.e. the one with UNK), without
# verbalized pronunciations.   This is the most common test setup, I understand.

#cp links/13-32.1/wsj1/doc/lng_modl/base_lm/bcb20onp.z $lmdir/lm_bg.arpa.gz || exit 1;
cp $wsj_lmdir/bcb20onp.z $lmdir/lm_bg.arpa.gz || exit 1;
chmod u+w $lmdir/lm_bg.arpa.gz

# trigram would be:
#cat links/13-32.1/wsj1/doc/lng_modl/base_lm/tcb20onp.z | \
# perl -e 'while(<>){ if(m/^\\data\\/){ print; last;  } } while(<>){ print; }' | \
# gzip -c -f > $lmdir/lm_tg.arpa.gz || exit 1;
#
#prune-lm --threshold=1e-7 $lmdir/lm_tg.arpa.gz $lmdir/lm_tgpr.arpa || exit 1;
#gzip -f $lmdir/lm_tgpr.arpa || exit 1;
#
## repeat for 5k language models
#cp links/13-32.1/wsj1/doc/lng_modl/base_lm/bcb05onp.z  $lmdir/lm_bg_5k.arpa.gz || exit 1;
cp $wsj_lmdir/bcb05onp.z  $lmdir/lm_bg_5k.arpa.gz || exit 1;
chmod u+w $lmdir/lm_bg_5k.arpa.gz
#
## trigram would be: !only closed vocabulary here!
#cp links/13-32.1/wsj1/doc/lng_modl/base_lm/tcb05cnp.z $lmdir/lm_tg_5k.arpa.gz || exit 1;
#chmod u+w $lmdir/lm_tg_5k.arpa.gz
#gunzip $lmdir/lm_tg_5k.arpa.gz
#tail -n 4328839 $lmdir/lm_tg_5k.arpa | gzip -c -f > $lmdir/lm_tg_5k.arpa.gz
#rm $lmdir/lm_tg_5k.arpa
#
#prune-lm --threshold=1e-7 $lmdir/lm_tg_5k.arpa.gz $lmdir/lm_tgpr_5k.arpa || exit 1;
#gzip -f $lmdir/lm_tgpr_5k.arpa || exit 1;
#

#if [ ! -f wsj0-train-spkrinfo.txt ] || [ `cat wsj0-train-spkrinfo.txt | wc -l` -ne 134 ]; then
#  rm wsj0-train-spkrinfo.txt
#  ! wget http://www.ldc.upenn.edu/Catalog/docs/LDC93S6A/wsj0-train-spkrinfo.txt && \
#    echo "Getting wsj0-train-spkrinfo.txt from backup location" && \
#    wget --no-check-certificate https://sourceforge.net/projects/kaldi/files/wsj0-train-spkrinfo.txt 
#fi
#
#if [ ! -f wsj0-train-spkrinfo.txt ]; then
#  echo "Could not get the spkrinfo.txt file from LDC website (moved)?"
#  echo "This is possibly omitted from the training disks; couldn't find it." 
#  echo "Everything else may have worked; we just may be missing gender info"
#  echo "which is only needed for VTLN-related diagnostics anyway."
#  exit 1
#fi
# Note: wsj0-train-spkrinfo.txt doesn't seem to be on the disks but the
# LDC put it on the web.  Perhaps it was accidentally omitted from the
# disks.  

#cat links/11-13.1/wsj0/doc/spkrinfo.txt \
#    links/13-32.1/wsj1/doc/evl_spok/spkrinfo.txt \
#    links/13-34.1/wsj1/doc/dev_spok/spkrinfo.txt \
#    links/13-34.1/wsj1/doc/train/spkrinfo.txt \
#   ./wsj0-train-spkrinfo.txt  | \
#    perl -ane 'tr/A-Z/a-z/; m/^;/ || print;' | \
#   awk '{print $1, $2}' | grep -v -- -- | sort | uniq > spk2gender


echo "Data preparation succeeded"

