#!/bin/bash

set -aux


MODEL_PATH="checkpoints/epoch_9_val_acc_0.554688"

export CUDA_VISIBLE_DEVICES=1
python3 run_infer.py \
  --model_path $MODEL_PATH
