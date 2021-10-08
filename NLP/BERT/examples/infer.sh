#!/bin/bash

MODEL_PATH="your convert model"

export CUDA_VISIBLE_DEVICES=1
python3 run_infer.py \
  --use_lazy_model \
  --model_path $MODEL_PATH
