#!/bin/bash
DEVICE_NUM_PER_NODE=1
DATA_DIR=/path/to/movielens_embdict
PERSISTENT_PATH=/path/to/persistent
MODEL_SAVE_DIR=/path/to/model/save/dir

python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    dssm_train_eval.py \
      --data_dir $DATA_DIR \
      --persistent_path $PERSISTENT_PATH \
      --table_size_array "16976, 23605, 49657" \
      --store_type 'cached_host_mem' \
      --cache_memory_budget_mb 1024 \
      --batch_size 4096 \
      --train_batches 75000 \
      --loss_print_interval 100 \
      --user_dnn_units "400, 400, 400" \
      --item_dnn_units "400, 400, 400" \
      --net_dropout 0.2 \
      --learning_rate 0.001 \
      --embedding_vec_size 10 \
      --num_train_samples 1404801 \
      --num_val_samples 401372 \
      --num_test_samples 200686 \
      --model_save_dir $MODEL_SAVE_DIR \
      --save_best_model
