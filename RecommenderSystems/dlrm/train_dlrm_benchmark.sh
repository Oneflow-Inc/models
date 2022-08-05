data_dir=${1}
ngpus=${2:-8}
pp=${3:-persistent}

export NCCL_CHECKS_DISABLE=1
export ONEFLOW_FUSE_MODEL_UPDATE_CAST=1
export ONEFLOW_ENABLE_MULTI_TENSOR_MODEL_UPDATE=1
export ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1
export ONEFLOW_EAGER_LOCAL_TO_GLOBAL_BALANCED_OVERRIDE=1
export ONEFLOW_ONE_EMBEDDING_FUSED_MLP_ASYNC_GRAD=1
export ONEFLOW_ONE_EMBEDDING_FUSE_EMBEDDING_INTERACTION=1
export ONEFLOW_EP_CUDA_DEVICE_FLAGS=2
export ONEFLOW_FUSE_BCE_REDUCE_MEAN_FW_BW=1

export ONEFLOW_ONE_EMBEDDING_ID_SHUFFLE_USE_P2P=1
export ONEFLOW_ONE_EMBEDDING_EMBEDDING_SHUFFLE_USE_P2P=1

# optional if batch_size = 55296
# export ONEFLOW_RAW_READER_FORCE_DIRECT_IO=1

python3 -m oneflow.distributed.launch \
    --nproc_per_node $ngpus \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    dlrm_benchmark_a100.py \
      --data_dir $data_dir \
      --persistent_path $pp \
      --amp

