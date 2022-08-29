persistent=./persistent
rm -rf ${prefix}.* $persistent/*

#export CUDA_VISIBLE_DEVICES=1
export ONEFLOW_FUSE_MODEL_UPDATE_CAST=1
export ONEFLOW_ENABLE_MULTI_TENSOR_MODEL_UPDATE=1
export ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1
export ONEFLOW_EAGER_LOCAL_TO_GLOBAL_BALANCED_OVERRIDE=1
#export ONEFLOW_ONE_EMBEDDING_ENABLE_QUANTIZED_COMM=1
export ONEFLOW_ONE_EMBEDDING_USE_SYSTEM_GATHER=0
#export ONEFLOW_ONE_EMBEDDING_EMBEDDING_SHUFFLE_INDEPENTENT_STREAM=1
export ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE=1

table_size_array='1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572'
num_train_examples=36672493
num_eval_samples=4584062
batch_size=512

train_batches=$(( num_train_examples / batch_size + 1 ))
eval_batches=$(( num_eval_samples / batch_size + 1 ))

python3 -m oneflow.distributed.launch \
    --nproc_per_node 1 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    dlrm_train_eval.py \
      --data_dir /data/criteo_kaggle/dlrm_parquet_int32 \
      --persistent_path $persistent \
      --store_type device_mem \
      --train_batches $train_batches \
      --train_batch_size $batch_size \
      --eval_batches $eval_batches \
      --eval_batch_size $batch_size \
      --learning_rate 0.2 \
      --table_size_array $table_size_array \
      --one_embedding_key_type int32 \
      --amp
