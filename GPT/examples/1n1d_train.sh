#! /bin/bash

export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED

# DATASET=/dataset/Megatron-LM/dummy/gpt_sample_dataset_text_document
DATASET=/dataset/gpt/gpt_sample_dataset_text_document

SRC_DIR=$(realpath $(dirname "$0")/..)

# gdb --args \
python3 $SRC_DIR/oneflow_gpt/train.py \
    --dataset $DATASET \
    --split 949,50,1 \
    --vocab-size 50257 \
    --seq-length 1024 \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --micro-batch-size 8 \
    --global-batch-size 8 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-gpus-per-node 1 \
    --num-nodes 1 \
    --train-iters 10 \
    --learning-rate 0.00015 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --lr-decay-iters 320000 \
    --lr-warmup-fraction 0.01 \
    --initial-loss-scale 4294967296 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --no-scale-tril-softmax-dropout-fusion \
    --fp16 \
    --graph \
