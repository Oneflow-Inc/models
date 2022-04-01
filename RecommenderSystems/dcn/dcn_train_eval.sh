#! /bin/sh
DEVICE_NUM_PER_NODE=1
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
DATA_DIR=/home/yuanziyang/yzywork/dcn-test-dir/Frappe_x1_parquet
PERSISTENT_PATH=./persistent


python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    dcn_train_eval.py \
      --data_dir $DATA_DIR \
      --persistent_path $PERSISTENT_PATH \
      --table_size_array "957, 3148, 7, 7, 2, 3, 2, 9, 80, 233" \
      --cache_memory_budget_mb 2048 \
      --train_batch_size 4096 \
      --train_batches 1000 \
      --loss_print_interval 2 \
      --eval_interval 2 \
      --eval_batch_size 256 \
      --eval_batches 113 \
      --warmup_batches 50 \
      --dnn_hidden_units "400,400,400" \
      --learning_rate 0.001 \
      --embedding_vec_size 10