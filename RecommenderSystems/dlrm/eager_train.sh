persistent=./persistent
rm -rf $persistent/*

#export CUDA_VISIBLE_DEVICES=1
export ONEFLOW_FUSE_MODEL_UPDATE_CAST=1
export ONEFLOW_ENABLE_MULTI_TENSOR_MODEL_UPDATE=1
export ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1
export ONEFLOW_EAGER_LOCAL_TO_GLOBAL_BALANCED_OVERRIDE=1
#export ONEFLOW_ONE_EMBEDDING_ENABLE_QUANTIZED_COMM=1
export ONEFLOW_ONE_EMBEDDING_USE_SYSTEM_GATHER=0
#export ONEFLOW_ONE_EMBEDDING_EMBEDDING_SHUFFLE_INDEPENTENT_STREAM=1


num_train_samples=4195197692
train_batch_size=55296
train_batch_size=512
warmup_batches=2500
decay_batches=15406

train_batches=$((num_train_samples / train_batch_size + 1))
decay_start=$((train_batches - decay_batches + 3700))
test_case=g${num_gpus}_lr${lr}_t${train_batches}_b${train_batch_size}_d${decay_batches}
echo $test_case

export ONEFLOW_ONE_EMBEDDING_EAGER=1
python3 -m oneflow.distributed.launch \
    --nproc_per_node 4 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    dlrm_eager_train_eval.py \
      --data_dir /RAID0/xiexuan/dlrm_parquet_int32 \
      --persistent_path $persistent \
      --eval_interval 10000 \
      --store_type cached_host_mem \
      --train_batches $train_batches \
      --train_batch_size $train_batch_size \
      --one_embedding_key_type int32 \
      --warmup_batches $warmup_batches \
      --decay_batches $decay_batches \
      --decay_start $decay_start \
      --amp | tee ${test_case}.log
