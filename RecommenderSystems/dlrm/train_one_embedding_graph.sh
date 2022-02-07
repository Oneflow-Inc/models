rm core.*
DEVICE_NUM_PER_NODE=4
MASTER_ADDR=127.0.0.1
NUM_NODES=1
NODE_RANK=0
DATA_DIR=/data/criteo1t/dlrm_v884m_ofrecord
EMBD_SIZE=33762578
BATHSIZE=55296
export GLOG_minloglevel=2
ulimit -SHn 131072
eval_batch_size=32744
eval_batchs=$(( 3274330 / eval_batch_size ))
#eval_batchs=$(( 90243072 / eval_batch_size ))

export LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc.so.4:
export BLOCK_BASED_PATH="rocks" 
echo "ll BLOCK_BASED_PATH"
ls -l $BLOCK_BASED_PATH
rm -rf rocks/0-4/*
rm -rf rocks/1-4/*
rm -rf rocks/2-4/*
rm -rf rocks/3-4/*

#/usr/local/cuda-11.4/bin/nsys profile --stat=true \
#numactl --interleave=all \
python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
  train.py \
    --interaction_type fused \
    --embedding_type OneEmbedding \
    --learning_rate 24 \
    --batch_size $BATHSIZE \
    --data_dir $DATA_DIR \
    --loss_print_every_n_iter 100 \
    --eval_interval 1000 \
    --eval_batchs $eval_batchs \
    --eval_batch_size $eval_batch_size \
    --max_iter 75868 \
    --vocab_size $EMBD_SIZE \
    --data_part_num 5888 \
    --data_part_name_suffix_length 5 \
    --execution_mode 'graph' \
    --cache_policy 'lru,none' \
    --cache_memory_budget_mb '16384,163840' \
    --value_memory_kind 'device,host' \
    --persistent_path $BLOCK_BASED_PATH \
    --column_size_array '227605432,39060,17295,7424,20265,3,7122,1543,63,130229467,3067956,405282,10,2209,11938,155,4,976,14,292775614,40790948,187188510,590152,12973,108,36' \
    --test_name 'train_one_embedding_graph_'$DEVICE_NUM_PER_NODE'gpu' | tee 'train_one_embedding_graph_'$DEVICE_NUM_PER_NODE'gpu'.log
    #--eval_save_dir '/NVME0/guoran/auc' \
    #--eval_after_training \
