#!/bin/bash

_DEVICE_NUM_PER_NODE=2
_MASTER_ADDR=127.0.0.1
_NUM_NODES=1
_NODE_RANK=0

# train_data_dir=/dataset/bert_regression_test/0
train_data_dir=/dataset/bert/of_wiki_seq_len_128
LOGFILE=./bert_graph_pretrain.log

# export ONEFLOW_DEBUG_MODE=1
export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES=1
python3 -m oneflow.distributed.launch \
    --nproc_per_node $_DEVICE_NUM_PER_NODE \
    --nnodes $_NUM_NODES \
    --node_rank $_NODE_RANK \
    --master_addr $_MASTER_ADDR \
    run_pretraining.py \
    --ofrecord_path $train_data_dir \
    --use_consistent \
    --metric-local 2>&1 | tee ${LOGFILE}