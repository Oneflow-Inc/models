#!/bin/bash
rm -r /home/yuanziyang/yzywork/dcn-test-dir/persistent1/*
rm -r /home/yuanziyang/yzywork/dcn-test-dir/persistent2/*

export CUDA_VISIBLE_DEVICES=0,1

DEVICE_NUM_PER_NODE=1
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
DATA_DIR=/home/yuanziyang/yzywork/dcn-test-dir/criteo_parquet
PERSISTENT_PATH=/home/yuanziyang/yzywork/dcn-test-dir/persistent1
PERSISTENT_PATH_FM=/home/yuanziyang/yzywork/dcn-test-dir/persistent2

python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    dcn_train_eval.py \
      --data_dir $DATA_DIR \
      --persistent_path $PERSISTENT_PATH \
      --persistent_path_fm $PERSISTENT_PATH_FM \
      --table_size_array "43, 98, 121, 41, 219, 112, 79, 68, 91, 5, 26, 36, 70, 1447, 554, 157461, 117683, 305, 17, 11878, 629, 3, 39504, 5128, 156729, 3175, 27, 11070, 149083, 10, 4542, 1996, 4, 154737, 17, 15, 52989, 81, 40882" \
      --store_type 'cached_host_mem' \
      --cache_memory_budget_mb 1024 \
      --train_batch_size 10000 \
      --train_batches 70000 \
      --loss_print_interval 100 \
      --eval_batch_size 10000 \
      --eval_batches 1000 \
      --dnn_hidden_units "1000, 1000, 1000, 1000, 1000" \
      --crossing_layers 4\
      --net_dropout 0.2 \
      --learning_rate 0.001 \
      --embedding_vec_size 16 \
      --disable_fusedmlp True \
      --num_train_samples 36672493 \
      --size_factor 3 > run.log 2>&1 & 
      tail -f run.log

      # --eval_batch_size 10000 \
      # --eval_batches 1000 \ 917