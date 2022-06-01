#!/bin/bash
rm -r /minio/sdb/sunbowen/persistent1/*

export CUDA_VISIBLE_DEVICES=2
export ONEFLOW_DEBUG_MODE=INFO
export ONEFLOW_EAGER_LOCAL_TO_GLOBAL_BALANCED_OVERRIDE=1

DEVICE_NUM_PER_NODE=1
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
DATA_DIR=/minio/sdb/sunbowen/dataset/criteo_x4/deepfm_parquet
PERSISTENT_PATH=/minio/sdb/sunbowen/persistent/persistent1

python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    xdeepfm_train_eval.py \
      --data_dir $DATA_DIR \
      --persistent_path $PERSISTENT_PATH \
      --table_size_array "649,9364,14746,490,476707,11618,4142,1373,7275,13,169,407,1376,1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572" \
      --store_type 'cached_host_mem' \
      --cache_memory_budget_mb 1024 \
      --batch_size 10000 \
      --train_batches 80000 \
      --loss_print_interval 100 \
      --dnn_hidden_units "1000,1000,1000,1000,1000" \
      --net_dropout 0.2 \
      --learning_rate 0.001 \
      --embedding_vec_size 16 \
      --num_train_samples 36672493 \
      --num_val_samples 4584062 \
      --num_test_samples 4584062 \
      --model_save_dir /minio/sdb/sunbowen/saved_models/of_deepfm \
      --save_best_model \
    > run.log & tail -f run.log
