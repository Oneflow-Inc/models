train_batch_size=${1:-55296}
rm -rf ./persistent/*

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
export LD_PRELOAD=/usr/lib64/libjemalloc.so.1
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NCCL_CHECKS_DISABLE=1
export ONEFLOW_EP_CUDA_STREAM_FLAGS=1
export ONEFLOW_EP_CUDA_DEVICE_FLAGS=2
export ONEFLOW_ONE_EMBEDDING_ADD_ID_SHUFFLE_COPY_OUT=1
#export ONEFLOW_DECODE_H2D_REGST_NUM=4
export ONEFLOW_RAW_READER_PREFETCHING_QUEUE_DEPTH=512
export ONEFLOW_RAW_READER_NUM_WORKERS=1
#export ONEFLOW_FUSE_BCE_REDUCE_MEAN_FW_BW=1
#export ONEFLOW_DEBUG_MODE=1

if [[ $train_batch_size -eq 55296 ]]; then
  export ONEFLOW_RAW_READER_FORCE_DIRECT_IO=1
fi

num_train_samples=4195197692
decay_batches=15406
train_batches=$((num_train_samples / train_batch_size + 1))
decay_start=$((train_batches - decay_batches + 3700))

#/usr/local/cuda-11.6/bin/nsys profile --stats=true \
numactl --interleave=all python3 -m oneflow.distributed.launch \
    --nproc_per_node 4 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    dlrm_benchmark_a100.py \
      --persistent_path ./persistent \
      --data_dir /RAID0/liujuncheng/raw \
      --store_type device_mem \
      --loss_print_interval 1000 \
      --one_embedding_key_type int32 \
      --eval_interval 100000 \
      --cache_memory_budget_mb 16384 \
      --train_batch_size $train_batch_size \
      --eval_batch_size 55296 \
      --train_batches $train_batches \
      --decay_batches $decay_batches \
      --decay_start $decay_start \
      --loss_scale_policy "static" \
      --split_allreduce \
      --table_size_array "39884407,39043,17289,7420,20263,3,7120,1543,63,38532952,2953546,403346,10,2208,11938,155,4,976,14,39979772,25641295,39664985,585935,12972,108,36"

