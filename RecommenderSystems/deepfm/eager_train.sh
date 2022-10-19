#!/bin/bash
export ONEFLOW_EAGER_LOCAL_TO_GLOBAL_BALANCED_OVERRIDE=1

DATA_DIR=/path/to/deepfm_parquet
PERSISTENT_PATH=/path/to/persistent
MODEL_SAVE_DIR=/path/to/model/save/dir
table_size_array="649,9364,14746,490,476707,11618,4142,1373,7275,13,169,407,1376,1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572"
device_mem=device_mem

rm -rf $PERSISTENT_PATH/*

export ONEFLOW_ONE_EMBEDDING_EAGER=1
python3 -m oneflow.distributed.launch \
    --nproc_per_node 4 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    deepfm_eager_train_eval.py \
      --data_dir $DATA_DIR \
      --persistent_path $PERSISTENT_PATH \
      --table_size_array $table_size_array \
      --store_type 'cached_host_mem' \
      --cache_memory_budget_mb 1024 \
      --batch_size 55296 \
      --dnn "1000,1000,1000,1000,1000" \
      --loss_print_interval 100 \
      --net_dropout 0.2 \
      --embedding_vec_size 16 \
      --num_train_samples 36672493 \
      --num_val_samples 4584062 \
      --num_test_samples 4584062
