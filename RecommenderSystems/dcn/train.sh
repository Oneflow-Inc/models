DEVICE_NUM_PER_NODE=4
DATA_DIR=/RAID0/xiexuan/criteo1t_parquet_40M_long
PERSISTENT_PATH=/home/zhengzekang/models_dcn/RecommenderSystems/dcn/init_model
MODEL_SAVE_DIR=/home/zhengzekang/models_dcn/RecommenderSystems/dcn/dcn_model

rm -rf /home/zhengzekang/models_dcn/RecommenderSystems/dcn/init_model/0-4/*
rm -rf /home/zhengzekang/models_dcn/RecommenderSystems/dcn/init_model/1-4/*
rm -rf /home/zhengzekang/models_dcn/RecommenderSystems/dcn/init_model/2-4/*
rm -rf /home/zhengzekang/models_dcn/RecommenderSystems/dcn/init_model/3-4/*


python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    dcn_train_eval.py \
      --data_dir $DATA_DIR \
      --model_save_dir $MODEL_SAVE_DIR \
      --persistent_path $PERSISTENT_PATH \
      --table_size_array "62866,8001,2901,74623,7530,3391,1400,21705,7937,21,276,1235896,9659,39884301,39040,17291,7421,20263,3,7121,1543,63,38532372,2953790,403302,10,2209,11938,155,4,976,14,39979538,25638302,39665755,585840,12973,108,36" \
      --store_type 'device_mem' \
      --dnn_hidden_units "1000, 1000, 1000, 1000, 1000" \
      --crossing_layers 4 \
      --num_train_samples 4195197692 \
      --num_test_samples 89137319 \
      --num_valid_samples 89137318 \
      --learning_rate 0.0025 \
      --train_batch_size 55296 \
      --train_batches 75000 \
      --net_dropout 0.0 \
      --embedding_vec_size 16 




