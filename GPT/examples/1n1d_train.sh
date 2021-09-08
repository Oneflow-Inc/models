#! /bin/bash

export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED

SRC_DIR=$(realpath $(dirname "$0")/..)

# gdb --args \
python3 $SRC_DIR/oneflow_gpt/train.py \
    --dataset /dataset/Megatron-LM/dummy/gpt_sample_dataset_text_document \
    --split 949,50,1 \
    --vocab-size 50257 \
    --seq-length 1024 \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-gpus-per-node 1 \
    --num-nodes 1 \
    --train-iters 10 \
    --lr-decay-iters 320000 \
    --no-scale-tril-softmax-dropout-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --graph \
