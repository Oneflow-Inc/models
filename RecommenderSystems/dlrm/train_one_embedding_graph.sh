rm core.*
DEVICE_NUM_PER_NODE=4
MASTER_ADDR=127.0.0.1
NUM_NODES=1
NODE_RANK=0
# DATA_DIR=/dataset/wdl_ofrecord/ofrecord
DATA_DIR=/data/xiexuan/criteo1t_ofrecord_shuffle_per_day/
EMBD_SIZE=33762578
BATHSIZE=55296
export GLOG_minloglevel=2
export CACHE_MEMORY_BUDGET_MB=16384 #8192
ulimit -SHn 131072

#export KEY_VALUE_STORE="cuda_in_memory"
#export NUM_KEYS=100000 #225000000 #209715200 #134217728
#export NUM_DEVICE_KEYS=1000 #25000000

export KEY_VALUE_STORE="block_based"
export BLOCK_BASED_PATH="/NVME0/guoran/rocks" #"init_model"
export LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc.so.4:

#/usr/local/cuda-11.4/bin/nsys profile --stat=true \
#numactl --interleave=all \
python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
  train.py \
    --interaction_type dot \
    --embedding_type OneEmbedding \
    --learning_rate 24 \
    --batch_size $BATHSIZE \
    --data_dir $DATA_DIR \
    --loss_print_every_n_iter 100 \
    --eval_interval 1000 \
    --max_iter 75868 \
    --vocab_size $EMBD_SIZE \
    --data_part_num 5888 \
    --data_part_name_suffix_length 5 \
    --execution_mode 'graph' \
    --test_name 'train_eager_graph_'$DEVICE_NUM_PER_NODE'gpu' | tee 'train_eager_graph_'$DEVICE_NUM_PER_NODE'gpu'.log
    #--dataset_format 'synthetic' \
