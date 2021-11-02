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
       --train-iters 1000 \
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
       # --fp16 \
"


source $1
DATESTR=$(date +"%m-%d-%H-%M")

_DEVICE_NUM_PER_NODE=1
_MASTER_ADDR=127.0.0.1
_NUM_NODES=1
_NODE_RANK=0

mkdir logs
run_cmd="python3 main_torch.py ${gpt_options} 2>&1 | tee logs/log-${DATESTR}.txt"

echo ${run_cmd}
eval ${run_cmd}

set +x
