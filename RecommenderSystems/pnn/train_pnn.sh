#!/bin/bash
rm -r /minio/sdb/sunbowen/persistent1/*

export CUDA_VISIBLE_DEVICES=1

DEVICE_NUM_PER_NODE=1
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
DATA_DIR=/minio/sdb/sunbowen/criteo_x4_9ea3bdfc/pnn_parquet
PERSISTENT_PATH=/minio/sdb/sunbowen/persistent1

python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    pnn_train_eval.py \
      --data_dir $DATA_DIR \
      --persistent_path $PERSISTENT_PATH \
      --table_size_array "43, 98, 121, 41, 219, 112, 79, 68, 91, 5, 26, 36, 70, 1447, 554, 157461, 117683, 305, 17, 11878, 629, 4, 39504, 5128, 156729, 3175, 27, 11070, 149083, 11, 4542, 1996, 4, 154737, 17, 16, 52989, 81, 40882" \
      --store_type 'cached_host_mem' \
      --cache_memory_budget_mb 1024 \
      --batch_size 2000 \
      --train_batches 75000 \
      --loss_print_interval 100 \
      --dnn "1000,1000" \
      --net_dropout 0.2 \
      --learning_rate 0.001 \
      --embedding_vec_size 16 \
      --num_train_samples 36672493 \
      --num_val_samples 4584062 \
      --num_test_samples 4584062 \
      --model_save_dir /minio/sdd/sunbowen/saved_models/of_pnn \
      --save_best_model \
      > run.log & tail -f run.log
