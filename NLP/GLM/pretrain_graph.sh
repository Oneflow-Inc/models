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
       --graph_fp16 \
"


source $1
DATESTR=$(date +"%m-%d-%H-%M")

_DEVICE_NUM_PER_NODE=1
_MASTER_ADDR=127.0.0.1
_NUM_NODES=1
_NODE_RANK=0

mkdir logs
run_cmd="python3 -m oneflow.distributed.launch \
    --nproc_per_node $_DEVICE_NUM_PER_NODE \
    --nnodes $_NUM_NODES \
    --node_rank $_NODE_RANK \
    --master_addr $_MASTER_ADDR \
    pretrain_glm_graph.py ${gpt_options} 2>&1 | tee logs/log-${DATESTR}.txt"

# env_cmd="export ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1 \
#         export ONEFLOW_THREAD_ENABLE_LOCAL_MESSAGE_QUEUE=1 \
#         export ONEFLOW_KERNEL_DISABLE_BLOB_ACCESS_CHECKER=1 \
#         export ONEFLOW_ACTOR_ENABLE_LIGHT_ACTOR=1 \
#         export ONEFLOW_STREAM_REUSE_CUDA_EVENT=1"

# echo ${env_cmd}
echo ${run_cmd}
eval ${run_cmd}

set +x
