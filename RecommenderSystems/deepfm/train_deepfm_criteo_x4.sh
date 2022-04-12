#!/bin/bash
rm -r /minio/sdd/sunbowen/data/persistent1/*
rm -r /minio/sdd/sunbowen/data/persistent2/*

export CUDA_VISIBLE_DEVICES=2

DEVICE_NUM_PER_NODE=1
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
DATA_DIR=/minio/sdd/sunbowen/criteo_sample/data/deepfm_parquet
PERSISTENT_PATH=/minio/sdd/sunbowen/data/persistent1
PERSISTENT_PATH_FM=/minio/sdd/sunbowen/data/persistent2

python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    deepfm_train_eval.py \
      --data_dir $DATA_DIR \
      --persistent_path $PERSISTENT_PATH \
      --persistent_path_fm $PERSISTENT_PATH_FM \
      --table_size_array "14, 68, 55, 35, 172, 92, 42, 41, 113, 4, 15, 5, 43, 27, 92, 172, 157, 12, 7, 183, 19, 2, 142, 173, 170, 166, 14, 170, 168, 9, 127, 44, 4, 169, 6, 10, 125, 20, 90" \
      --store_type 'cached_host_mem' \
      --cache_memory_budget_mb 1024 \
      --train_batch_size 10000 \
      --train_batches 35000 \
      --loss_print_interval 100 \
      --eval_batch_size 10000 \
      --eval_batches 1000 \
      --dnn "1000,1000,1000,1000,1000" \
      --net_dropout 0.2 \
      --learning_rate 0.001 \
      --embedding_vec_size 16 \
      --disable_fusedmlp True \
      --max_gradient_norm 1.0 \
      --num_train_samples 36672493 \
      > run.log & tail -f run.log
