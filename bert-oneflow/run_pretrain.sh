#!/bin/bash

train_data_dir=/dataset/bert_regression_test/0
LOGFILE=./bert_pretrain.log

# export ONEFLOW_DEBUG_MODE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
python3 run_pretraining.py \
    --ofrecord_path $train_data_dir 2>&1 | tee ${LOGFILE}