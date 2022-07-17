#!/bin/bash
rm -rf /home/sunbowen/one/models/RecommenderSystems/din/persistent/*

export CUDA_VISIBLE_DEVICES=0
export ONEFLOW_DEBUG_MODE=INFO
export ONEFLOW_EAGER_LOCAL_TO_GLOBAL_BALANCED_OVERRIDE=1
export OMP_NUM_THREADS=1
DEVICE_NUM_PER_NODE=1
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
DATA_DIR=/home/sunbowen/din/data
PERSISTENT_PATH=/home/sunbowen/one/models/RecommenderSystems/din/persistent/persistent
MODEL_LOAD_DIR=/home/sunbowen/one/models/RecommenderSystems/din/models/0/rec/with_emb
MODEL_SAVE_DIR=/home/sunbowen/din/saved_models/of_deepfm
python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    din_train_eval.py \
      --data_dir $DATA_DIR \
      --persistent_path $PERSISTENT_PATH \
      --id_table_size_array "63001" \
      --cat_table_size_array "801" \
      --store_type 'cached_host_mem' \
      --cache_memory_budget_mb 64 \
      --batch_size 2 \
      --train_batches 32 \
      --loss_print_interval 1 \
      --attention_layer_hidden_dim "80,40" \
      --net_dropout 0 \
      --learning_rate 0.85 \
      --optim "SGD" \
      --patience 8 \
      --item_emb_size 64 \
      --cat_emb_size 64 \
      --num_train_samples 64 \
      --num_val_samples 64 \
      --num_test_samples 64 \
      --model_load_dir $MODEL_LOAD_DIR \
      --model_save_dir $MODEL_SAVE_DIR \
      --save_best_model \
    > run.log & tail -f run.log

