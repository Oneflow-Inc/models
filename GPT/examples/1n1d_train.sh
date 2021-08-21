#! /bin/bash

src_dir=$(realpath $(dirname "$0")/..)

python3 $src_dir/oneflow_gpt/training.py \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --micro-batch-size 8 \
    --global-batch-size 8 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-gpus-per-node 1 \
    --num-nodes 1 \
    --node-ips 192.168.1.16 \
    --train-iters 10 \
    --dataset /dataset/Megatron-LM/dummy/gpt_sample_dataset_text_document \
    --seq-length 1024 \
    --vocab-size 50257 \
    --split 949,50,1 \
    --no-scale-tril-softmax-dropout-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
