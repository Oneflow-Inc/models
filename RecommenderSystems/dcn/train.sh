DEVICE_NUM_PER_NODE=1
DATA_DIR=your_path/criteo_parquet
PERSISTENT_PATH=your_path/persistent1
MODEL_SAVE_DIR=your_path/model_save_dir

python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    dcn_train_eval.py \
      --data_dir $DATA_DIR \
      --model_save_dir $MODEL_SAVE_DIR \
      --persistent_path $PERSISTENT_PATH \
      --table_size_array "649,9364,14746,490,476707,11618,4142,1373,7275,13,169,407,1376,1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572" \
      --store_type 'cached_host_mem' \
      --cache_memory_budget_mb 2048 \
      --dnn_hidden_units "1000, 1000, 1000, 1000, 1000" \
      --crossing_layers 4 \
      --embedding_vec_size 16 




