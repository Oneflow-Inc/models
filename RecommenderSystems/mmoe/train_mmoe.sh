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
      --table_size_array "8, 38, 36, 16, 3, 21, 14, 5, 8, 2, 3, 5, 7, 6, 5, 16, 13, 7, 6, 8, 8, 3, 4, 4, 15, 16, 14, 5, 3" \
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
