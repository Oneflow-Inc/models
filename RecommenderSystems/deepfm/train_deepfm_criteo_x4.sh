#!/bin/bash
rm -r /minio/sdb/liuxinman/persistent1/*
rm -r /minio/sdb/liuxinman/persistent2/*

export CUDA_VISIBLE_DEVICES=2,3

DEVICE_NUM_PER_NODE=1
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
DATA_DIR=/minio/sdb/liuxinman/criteo_x4/deepfm_parquet
PERSISTENT_PATH=/minio/sdb/liuxinman/persistent1
PERSISTENT_PATH_FM=/minio/sdb/liuxinman/persistent2

python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    deepfm_train_eval.py \
      --data_dir $DATA_DIR \
      --persistent_path $PERSISTENT_PATH \
      --persistent_path_fm $PERSISTENT_PATH_FM \
      --table_size_array "648, 9364, 14745, 489, 476706, 11617, 4141, 1372, 7274, 12, 168, 406, 1375, 1460, 583, 8959674, 2143424, 305, 24, 12516, 633, 3, 93051, 5682, 7544449, 3193, 27, 14989, 5107195, 10, 5652, 2173, 4, 6466377, 18, 15, 285147, 105, 142348" \
      --store_type 'cached_host_mem' \
      --cache_memory_budget_mb 1024 \
      --train_batch_size 10000 \
      --train_batches 35000 \
      --loss_print_interval 100 \
      --eval_interval 5000 \
      --eval_batch_size 10000 \
      --eval_batches 1000 \
      --dnn "1000,1000,1000,1000,1000" \
      --learning_rate 0.001 \
      --embedding_vec_size 16 \
      --disable_fusedmlp False \
      --weight_decay 1.0e-5 \
      --num_train_samples 36672493 \
      > run.log & tail -f run.log
