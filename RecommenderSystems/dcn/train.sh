export CUDA_VISIBLE_DEVICES=0

DEVICE_NUM_PER_NODE=1
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
DATA_DIR=your_path/criteo_parquet
PERSISTENT_PATH=your_path/persistent1
MODEL_SAVE_DIR=your_path/model_save_dir

python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    dcn_train_eval.py \
      --data_dir $DATA_DIR \
      --model_save_dir $MODEL_SAVE_DIR \
      --persistent_path $PERSISTENT_PATH \
      --save_initial_model \
      --save_model_after_each_eval \
      --table_size_array "43, 98, 121, 41, 219, 112, 79, 68, 91, 5, 26, 36, 70, 1447, 554, 157461, 117683, 305, 17, 11878, 629, 4, 39504, 5128, 156729, 3175, 27, 11070, 149083, 11, 4542, 1996, 4, 154737, 17, 16, 52989, 81, 40882" \
      --store_type 'cached_host_mem' \
      --cache_memory_budget_mb 2048 \
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
      --num_train_samples 36672493 \
      --size_factor 3 > run.log 2>&1 & 
      tail -f run.log
