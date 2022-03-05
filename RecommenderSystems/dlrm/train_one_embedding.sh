rm core.*
DEVICE_NUM_PER_NODE=${1-1}

MASTER_ADDR=127.0.0.1
NUM_NODES=1
NODE_RANK=0
DATA_DIR=/minio/sdd/dataset/criteo1t/add_slot_size_snappy_true
column_size_array='227605432,39060,17295,7424,20265,3,7122,1543,63,130229467,3067956,405282,10,2209,11938,155,4,976,14,292775614,40790948,187188510,590152,12973,108,36'
# eval_batch_size=65536
# eval_batch_size=524288 OOM@14
eval_batch_size=8192
# eval_batchs=$(( 89137319 / eval_batch_size ))
# eval_batch_size=65536
eval_batchs=10

EMBD_SIZE=33762577 # 33762578
BATHSIZE=8192

emb_size=16
# export CUDA_VISIBLE_DEVICES=1
export ONEFLOW_DEBUG_MODE=True
export EMBEDDING_SIZE=$emb_size

block_based_dir=/minio/sde/model_zoo/dlrm_baseline_params_emb$emb_size

cp $block_based_dir/one_embedding0/index_cp $block_based_dir/one_embedding0/index

#value_memory_kind: device, host
#cache_policy: lru, full, none?
cache_type="device_host" # "device_host"
if [ "$cache_type" = "device_ssd" ]; then
    # gpu + ssd
    cache_policy="lru"
    cache_memory_budget_mb="16384"
    value_memory_kind="device"
elif [ "$cache_type" = "device_only" ]; then 
    # gpu only, cache_memory > embedding table
    cache_policy="full"
    cache_memory_budget_mb="16384"
    value_memory_kind="device"
elif [ "$cache_type" = "host_ssd" ]; then 
    # cpu only, cache_memory > embedding table
    cache_policy="lru"
    cache_memory_budget_mb="16384"
    value_memory_kind="host"
elif [ "$cache_type" = "host_only" ]; then 
    # cpu only, cache_memory > embedding table
    cache_policy="full"
    cache_memory_budget_mb="16384"
    value_memory_kind="host"
elif [ "$cache_type" = "device_host_ssd" ]; then 
    # gpu + cpu + ssd
    cache_policy="lru,lru"
    cache_memory_budget_mb="1024,2048"
    value_memory_kind="device,host"    
else
    # gpu + cpu, fastest
    cache_policy="lru,full"
    cache_memory_budget_mb="1024,16384"
    value_memory_kind="device,host"
fi

python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
  graph_train.py \
    --model_save_dir ckpt \
    --column_size_array $column_size_array \
    --embedding_type OneEmbedding \
    --bottom_mlp 512,256,$emb_size \
    --top_mlp 1024,1024,512,256 \
    --embedding_vec_size $emb_size \
    --embedding_split_axis 1 \
    --learning_rate 2 \
    --batch_size $BATHSIZE \
    --data_dir $DATA_DIR \
    --loss_print_every_n_iter 100 \
    --eval_interval 100 \
    --eval_batchs $eval_batchs \
    --eval_batch_size $eval_batch_size \
    --max_iter 2000 \
    --vocab_size $EMBD_SIZE \
    --data_part_num 256 \
    --data_part_name_suffix_length 5 \
    --persistent_path $block_based_dir/one_embedding0 \
    --cache_policy $cache_policy \
    --cache_memory_budget_mb $cache_memory_budget_mb \
    --value_memory_kind $value_memory_kind \
    --test_name train_${DEVICE_NUM_PER_NODE}gpu

    # --save_init \
    # --save_model_after_each_eval \
    # --eval_after_training \
    # --model_load_dir /tank/model_zoo/dlrm_baseline_params_emb$emb_size \
    # --model_load_dir /tank/xiexuan/dlrm/initial_parameters \
