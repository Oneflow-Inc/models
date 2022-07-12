#!/bin/bash
#rm -rf /minio/sdb/sunbowen/persistent/*
rm -rf /home/sunbowen/din/persistent/*

export CUDA_VISIBLE_DEVICES=1
export ONEFLOW_DEBUG_MODE=INFO
export ONEFLOW_EAGER_LOCAL_TO_GLOBAL_BALANCED_OVERRIDE=1
export OMP_NUM_THREADS=1
DEVICE_NUM_PER_NODE=1
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
DATA_DIR=/home/sunbowen/din/data_big
PERSISTENT_PATH=/home/sunbowen/din/persistent/persistent
MODEL_LOAD_DIR=/home/sunbowen/din/init_models/
MODEL_SAVE_DIR=/home/sunbowen/din/saved_models/of_deepfm
python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    din_train_eval.py \
      --data_dir $DATA_DIR \
      --persistent_path $PERSISTENT_PATH \
      --id_table_size_array "63001" \
      --cat_table_size_array "801" \
      --store_type 'cached_host_mem' \
      --cache_memory_budget_mb 64 \
      --batch_size 32 \
      --train_batches 733715 \
      --loss_print_interval 100 \
      --attention_layer_hidden_dim "80,40" \
      --net_dropout 0 \
      --learning_rate 0.85 \
      --embedding_vec_size 64 \
      --num_train_samples 2608764 \
      --num_val_samples 384806 \
      --num_test_samples 384806 \
      --model_load_dir $MODEL_LOAD_DIR \
      --model_save_dir $MODEL_SAVE_DIR \
      --save_best_model \
    > run.log & tail -f run.log
