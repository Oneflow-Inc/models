rm -rf ./persistent/*

export NCCL_CHECKS_DISABLE=1
export ONEFLOW_FUSE_MODEL_UPDATE_CAST=1
export ONEFLOW_ENABLE_MULTI_TENSOR_MODEL_UPDATE=1
export ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1
export ONEFLOW_EAGER_LOCAL_TO_GLOBAL_BALANCED_OVERRIDE=1
export ONEFLOW_ONE_EMBEDDING_FUSED_MLP_ASYNC_GRAD=1
export ONEFLOW_ONE_EMBEDDING_FUSE_EMBEDDING_INTERACTION=1
export ONEFLOW_EP_CUDA_DEVICE_FLAGS=2
export ONEFLOW_FUSE_BCE_REDUCE_MEAN_FW_BW=1
export ONEFLOW_RAW_READER_FORCE_DIRECT_IO=1

export ONEFLOW_ONE_EMBEDDING_ID_SHUFFLE_USE_P2P=1
export ONEFLOW_ONE_EMBEDDING_EMBEDDING_SHUFFLE_USE_P2P=1

export CUDA_DEVICE_MAX_CONNECTIONS=32
export ONEFLOW_EP_CUDA_STREAM_FLAGS=1
export ONEFLOW_RAW_READER_PREFETCHING_QUEUE_DEPTH=512
export ONEFLOW_RAW_READER_NUM_WORKERS=1

export LD_PRELOAD=/usr/lib64/libjemalloc.so.1

numactl --interleave=all \
python3 -m oneflow.distributed.launch \
    --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    dlrm_benchmark_a100.py \
      --persistent_path ./persistent \
      --data_dir /RAID0/criteo1t_oneflow_raw \
      --split_allreduce \
      --amp

