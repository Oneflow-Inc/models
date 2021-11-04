#! /bin/bash

gpt_options=" \
       --block-lm \
       --task-mask \
       --bert-prob 0.5 \
       --gap-sentence-prob 0.3 \
       --avg-block-length 3 \
       --gpt-min-ratio 0.25 \
       --experiment-name blocklm-large-blank \
       --num-layers 12 \
       --hidden-size 512 \
       --num-attention-heads 8 \
       --seq-length 256 \
       --max-position-embeddings 256 \
       --save other/checkpoints \
       --train-iters 125 \
       --resume-dataloader \
       --train-data bert-large \
       --tokenizer-type BertWordPieceTokenizer \
       --tokenizer-model-type bert-large-uncased \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --lr-decay-iters 160000 \
       --lr-decay-ratio 0.05 \
       --warmup .05 \
       --num-workers 1\
       --ddp \
       # --fp16 \
"


source $1
DATESTR=$(date +"%m-%d-%H-%M")

_DEVICE_NUM_PER_NODE=8
_MASTER_ADDR=10.0.22.16
_NUM_NODES=2
_NODE_RANK=0
_MASTER_PORT=8089

mkdir logs
run_cmd="python3 -m oneflow.distributed.launch \
    --nproc_per_node $_DEVICE_NUM_PER_NODE \
    --nnodes $_NUM_NODES \
    --node_rank $_NODE_RANK \
    --master_addr $_MASTER_ADDR \
    --master_port $_MASTER_PORT \
    pretrain_glm.py ${gpt_options} 2>&1 | tee logs/log-${DATESTR}.txt"

echo ${run_cmd}
eval ${run_cmd}

set +x
