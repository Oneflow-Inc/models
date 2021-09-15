#!/bin/bash

set -aux

OFRECORD_PATH="sample_seq_len_512_example"
#if [ ! -d "$OFRECORD_PATH" ]; then
#    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/wiki_ofrecord_seq_len_128_example.tgz
#    tar zxf wiki_ofrecord_seq_len_128_example.tgz
#fi

CHECKPOINT_PATH="checkpoints"
if [ ! -d "$CHECKPOINT_PATH" ]; then
    mkdir $CHECKPOINT_PATH
fi

LEARNING_RATE=1e-4
EPOCH=10
TRAIN_BATCH_SIZE=8
VAL_BATCH_SIZE=8

#export CUDA_VISIBLE_DEVICES=1
DATA_DIR=wiki_ofrecord_seq_len_128_example
python3 run_pretraining.py \
  --ofrecord_path $OFRECORD_PATH \
  --checkpoint_path $CHECKPOINT_PATH \
  --lr $LEARNING_RATE \
  --epochs $EPOCH \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --val_batch_size $VAL_BATCH_SIZE \
  --seq_length=512 \
  --max_predictions_per_seq=80 \
  --num_hidden_layers=12 \
  --num_attention_heads=12 \
  --max_position_embeddings=512 \
  --type_vocab_size=2 \
  --vocab_size=30522 \
  --attention_probs_dropout_prob=0.1 \
  --hidden_dropout_prob=0.1 \
  --hidden_size_per_head=64
