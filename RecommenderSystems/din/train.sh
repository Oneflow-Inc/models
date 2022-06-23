#!/bin/bash
rm -r /minio/sdb/sunbowen/persistent/*

export ONEFLOW_DEBUG_MODE=INFO
export ONEFLOW_EAGER_LOCAL_TO_GLOBAL_BALANCED_OVERRIDE=1

DEVICE_NUM_PER_NODE=1
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
DATA_DIR=/minio/sdd/sunbowen/din/data
PERSISTENT_PATH=/minio/sdb/sunbowen/persistent/persistent1
MODEL_SAVE_DIR=/minio/sdb/sunbowen/saved_models/of_deepfm
python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    din_train_eval.py \
      --data_dir $DATA_DIR \
      --persistent_path $PERSISTENT_PATH \
      --table_size_array "64000" \
      --store_type 'cached_host_mem' \
      --cache_memory_budget_mb 1024 \
      --batch_size 200 \
      --train_batches 652191 \
      --loss_print_interval 100 \
      --attention_layer_hidden_dim "80,40" \
      --net_dropout 0.2 \
      --learning_rate 0.001 \
      --embedding_vec_size 64 \
      --num_train_samples 2608764 \
      --num_val_samples 384806 \
      --num_test_samples 384806 \
      --model_save_dir $MODEL_SAVE_DIR \
      --save_best_model \
    > run.log & tail -f run.log
