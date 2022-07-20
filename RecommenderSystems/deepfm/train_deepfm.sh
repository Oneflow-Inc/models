#!/bin/bash
DEVICE_NUM_PER_NODE=1
export ONEFLOW_EAGER_LOCAL_TO_GLOBAL_BALANCED_OVERRIDE=1

DATA_DIR=/path/to/deepfm_parquet
PERSISTENT_PATH=/path/to/persistent
rm -rf $PERSISTENT_PATH/*

table_size_array="39884406,39043,17289,7420,20263,3,7120,1543,63,38532951,2953546,403346,10,2208,11938,155,4,976,14,39979771,25641295,39664984,585935,12972,108,36"
num_train_samples=4195197692
num_eval_examples=89137319
batch_size=$(( 6912 * DEVICE_NUM_PER_NODE ))
train_batches=$(( num_train_samples / batch_size + 1 ))

python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    deepfm_train_eval.py \
      --data_dir $DATA_DIR \
      --persistent_path $PERSISTENT_PATH \
      --table_size_array $table_size_array \
      --store_type cached_host_mem \
      --cache_memory_budget_mb 2048 \
      --loss_print_interval 1000 \
      --net_dropout 0.5 \
      --feature_vec_size 10 \
      --train_batches $train_batches \
      --batch_size $batch_size \
      --num_train_samples $num_train_samples \
      --num_val_samples $num_eval_examples \
      --num_test_samples 0 \
      --amp
