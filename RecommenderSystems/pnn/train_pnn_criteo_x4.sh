#!/bin/bash
rm -r /minio/sdb/sunbowen/persistent1/*

export CUDA_VISIBLE_DEVICES=3

DEVICE_NUM_PER_NODE=1
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
DATA_DIR=/minio/sdb/sunbowen/criteo_x4_9ea3bdfc/deepfm_parquet
PERSISTENT_PATH=/minio/sdb/sunbowen/persistent1

python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    pnn_train_eval.py \
      --data_dir $DATA_DIR \
      --persistent_path $PERSISTENT_PATH \
      --table_size_array "36, 91, 112, 29, 211, 94, 70, 46, 83, 5, 24, 30, 42, 799, 544, 22916, 25358, 200, 13, 10126, 383, 3, 16872, 4508, 23375, 3106, 27, 7077, 24371, 10, 3307, 1628, 4, 23796, 13, 15, 15846, 61, 12817" \
      --store_type 'cached_host_mem' \
      --cache_memory_budget_mb 1024 \
      --train_batch_size 10000 \
      --train_batches 35000 \
      --loss_print_interval 100 \
      --eval_batch_size 10000 \
      --eval_batches 1000 \
      --dnn "1000,1000" \
      --net_dropout 0.2 \
      --learning_rate 0.001 \
      --embedding_vec_size 16 \
      --disable_fusedmlp True \
      --num_train_samples 36672493 \
      > run.log & tail -f run.log
