#!/bin/bash
DEVICE_NUM_PER_NODE=8
DATA_DIR=/RAID0/criteo1t_oneflow_raw
PERSISTENT_PATH=/RAID0/persistent

rm -rf $PERSISTENT_PATH/*
rm -rf rocks

export ONEFLOW_FUSE_MODEL_UPDATE_CAST=1
export ONEFLOW_ENABLE_MULTI_TENSOR_MODEL_UPDATE=1
export ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1
export ONEFLOW_EAGER_LOCAL_TO_GLOBAL_BALANCED_OVERRIDE=1
export ONEFLOW_ONE_EMBEDDING_FUSED_MLP_ASYNC_GRAD=1
export ONEFLOW_ONE_EMBEDDING_FUSE_EMBEDDING_INTERACTION=1
export ONEFLOW_ONE_EMBEDDING_EMBEDDING_GRADIENT_SHUFFLE_INDEPENTENT_STREAM=1
#export ONEFLOW_ONE_EMBEDDING_ID_SHUFFLE_USE_P2P=1
#export ONEFLOW_ONE_EMBEDDING_EMBEDDING_SHUFFLE_USE_P2P=1
#export ONEFLOW_ONE_EMBEDDING_EMBEDDING_GRADIENT_SHUFFLE_USE_P2P=0
#export PYTHONPATH=
#export PYTHONPATH=/root/guoran/oneflow/python/
export LD_PRELOAD=/usr/lib64/libjemalloc.so.1
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NCCL_CHECKS_DISABLE=1
export ONEFLOW_EP_CUDA_STREAM_FLAGS=1
export ONEFLOW_EP_CUDA_DEVICE_FLAGS=2
export ONEFLOW_ONE_EMBEDDING_ADD_ID_SHUFFLE_COPY_OUT=1
#export ONEFLOW_DECODE_H2D_REGST_NUM=4
export ONEFLOW_RAW_READER_PREFETCHING_QUEUE_DEPTH=512
export ONEFLOW_RAW_READER_NUM_WORKERS=1
export ONEFLOW_RAW_READER_FORCE_DIRECT_IO=1
#export ONEFLOW_FUSE_BCE_REDUCE_MEAN_FW_BW=1
#export ONEFLOW_DEBUG_MODE=1

# /usr/local/cuda-11.4/nsight-systems-2021.2.4/target-linux-x64/nsys profile --stat=true --force-overwrite true \
# --output="oneflow_dcn_1n4d_55296_fp32" \
python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    dcn_train_eval_raw.py \
      --data_dir $DATA_DIR \
      --persistent_path $PERSISTENT_PATH \
      --table_size_array "62866,8001,2901,74623,7530,3391,1400,21705,7937,21,276,1235896,9659,39884301,39040,17291,7421,20263,3,7121,1543,63,38532372,2953790,403302,10,2209,11938,155,4,976,14,39979538,25638302,39665755,585840,12973,108,36" \
      --store_type 'device_mem' \
      --train_batch_size 55296 \
      --train_batches 75000 \
      --test_batches 1612 \
      --test_batch_size 55296 \
      --loss_print_interval 1000 \
      --dnn_hidden_units "1000, 1000, 1000, 1000, 1000" \
      --crossing_layers 4 \
      --learning_rate 0.001 \
      --embedding_vec_size 16 \
      --num_train_samples 4195197692 \
      --num_test_samples 89137319 \
      --net_dropout 0.05

