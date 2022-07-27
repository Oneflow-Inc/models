#!/bin/bash
DEVICE_NUM_PER_NODE=4
DATA_DIR=/RAID0/xiexuan/criteo1t_parquet_40M_long
PERSISTENT_PATH=/home/zhengzekang/models/RecommenderSystems/dlrm/init_model


rm -rf /home/zhengzekang/models/RecommenderSystems/dlrm/init_model/0-4/*
rm -rf /home/zhengzekang/models/RecommenderSystems/dlrm/init_model/1-4/*
rm -rf /home/zhengzekang/models/RecommenderSystems/dlrm/init_model/2-4/*
rm -rf /home/zhengzekang/models/RecommenderSystems/dlrm/init_model/3-4/*

export ONEFLOW_ONE_EMBEDDING_FUSED_MLP_ASYNC_GRAD=1 

export ONEFLOW_ONE_EMBEDDING_FUSE_EMBEDDING_INTERACTION=1 
export ONEFLOW_ONE_EMBEDDING_GRADIENT_SHUFFLE_USE_FP16=1 
export ONEFLOW_FUSE_MODEL_UPDATE_CAST=1 
export ONEFLOW_ENABLE_MULTI_TENSOR_MODEL_UPDATE=1 
export ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1 
export ONEFLOW_EAGER_LOCAL_TO_GLOBAL_BALANCED_OVERRIDE=1 
export ONEFLOW_ONE_EMBEDDING_USE_SYSTEM_GATHER=0
export ONEFLOW_EP_CUDA_DEVICE_SCHEDULE=2
export ONEFLOW_EP_CUDA_STREAM_NON_BLOCKING=1
export ONEFLOW_ONE_EMBEDDING_ADD_ID_SHUFFLE_COPY_OUT=1
export ONEFLOW_ONE_EMBEDDING_FUSE_EMBEDDING_INTERACTION=1


export ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE=1

python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    dcn_train_eval.py \
      --data_dir $DATA_DIR \
      --persistent_path $PERSISTENT_PATH \
      --table_size_array "62866,8001,2901,74623,7530,3391,1400,21705,7937,21,276,1235896,9659,39884301,39040,17291,7421,20263,3,7121,1543,63,38532372,2953790,403302,10,2209,11938,155,4,976,14,39979538,25638302,39665755,585840,12973,108,36" \
      --store_type 'device_mem' \
      --train_batch_size 55296 \
      --train_batches 75000 \
      --loss_print_interval 1000 \
      --dnn_hidden_units "1000, 1000, 1000, 1000, 1000" \
      --crossing_layers 4 \
      --learning_rate 0.003 \
      --embedding_vec_size 16 \
      --num_train_samples 4195197692 \
      --num_valid_samples 89137318 \
      --num_test_samples 89137319 \
      --net_dropout 0.05 \
      --amp