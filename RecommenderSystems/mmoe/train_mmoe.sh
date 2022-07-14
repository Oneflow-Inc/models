#!/bin/bash
DEVICE_NUM_PER_NODE=1
DATA_DIR=/path/to/mmoe_parquet
PERSISTENT_PATH=/path/to/persistent
MODEL_SAVE_DIR=/path/to/model/save/dir

python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    mmoe_train_eval.py \
      --data_dir $DATA_DIR \
      --persistent_path $PERSISTENT_PATH \
      --table_size_array "9, 52, 47, 17, 3, 24, 15, 5, 10, 2, 3, 6, 8, 6, 6, 51, 38, 8, 10, 9, 10, 3, 4, 5, 43, 43, 43, 5, 3" \
      --store_type 'cached_host_mem' \
      --cache_memory_budget_mb 1024 \
      --batch_size 256 \
      --train_batches 16000 \
      --loss_print_interval 100 \
      --learning_rate 0.001 \
      --embedding_vec_size 4 \
      --expert_dnn "256, 128" \
      --num_train_samples 199523 \
      --num_test_samples 99762 \
      --model_save_dir $MODEL_SAVE_DIR
