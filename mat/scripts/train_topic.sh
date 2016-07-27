#!/bin/sh

CASE=$1
dir=$(pwd)
corpus=$dir/exp/merge/corpus.txt
binary=$dir/exp/merge/corpus.bin

TOPIC=$1
ITERATION=1000
MODEL=$dir/exp/$CASE/model
INFER=$dir/exp/$CASE/inference
ALPHA=1

#cd mallet

#bin/mallet import-file --input $corpus --output $binary --keep-sequence
#bin/mallet train-topics --input $binary --num-topics $TOPIC --num-iterations $ITERATION --output-model $MODEL --output-doc-topics $INFER --alpha $ALPHA --beta 0

#cd ..


mallet import-file --input $corpus --output $binary --keep-sequence
mallet train-topics --input $binary --num-topics $TOPIC --num-iterations $ITERATION --output-model $MODEL --output-doc-topics $INFER --alpha $ALPHA --beta 0
