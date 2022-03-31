DEVICE_NUM_PER_NODE=1
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
DATA_DIR=/minio/sdb/liuxinman/criteo_sample/deepfm_parquet
PERSISTENT_PATH=/minio/sdb/liuxinman/persistent

python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    deepfm_train_eval.py \
      --data_dir $DATA_DIR \
      --persistent_path $PERSISTENT_PATH \
      --table_size_array "14, 68, 55, 35, 172, 92, 42, 41, 113, 4, 15, 5, 43, 27, 92, 172, 157, 12, 7, 183, 19, 2, 142, 173, 170, 166, 14, 170, 168, 9, 127, 44, 4, 169, 6, 10, 125, 20, 90" \
      --cache_memory_budget_mb 2048 \
      --train_batch_size 2 \
      --train_batches 100 \
      --loss_print_interval 2 \
      --eval_interval 20 \
      --eval_batch_size 5 \
      --eval_batches 3 \
      --warmup_batches 10 \
      --dnn "512,512,256,128" \
      --learning_rate 0.001 \
      --embedding_vec_size 5