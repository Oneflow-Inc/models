#!/bin/bash
rm -rf ./persistent/*
#cp -r /data/xiexuan/model_zoo/of_din_init_ckpt/persistent/* ./persistent

export ONEFLOW_DEBUG_MODE=1
#export CUDA_VISIBLE_DEVICES=2
DEVICE_NUM_PER_NODE=4
DATA_DIR=/data/xiexuan/git-repos/models/RecommenderSystems/din/amazon_elec_parquet

python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    din_train_eval_pq.py \
      --data_dir $DATA_DIR \
      --batch_size 1024 \
      --max_len 512 \
      --train_batches 1630490 \
      --learning_rate 8.0 \
      | tee run.log
      #--model_load_dir /data/xiexuan/model_zoo/of_din_init_ckpt/merged_tf \
      #--save_model_after_each_eval \
      #--model_save_dir ckpt \
      # --save_initial_model \
