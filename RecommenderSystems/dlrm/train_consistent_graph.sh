rm core.*
DEVICE_NUM_PER_NODE=4
MASTER_ADDR=127.0.0.1
NUM_NODES=1
NODE_RANK=0
# DATA_DIR=/dataset/wdl_ofrecord/ofrecord
DATA_DIR=/data/xiexuan/criteo1t_ofrecord_shuffle_per_day/
EMBD_SIZE=33762578
BATHSIZE=65536
#export GLOG_minloglevel=2
ulimit -SHn 131072
export emb_size=128

eval_batch_size=327432
eval_batchs=$(( 3274330 / eval_batch_size ))

export L1_CACHE_POLICY="lru"
export L1_CACHE_MEMORY_BUDGET_MB=16384 #8192
export L2_CACHE_POLICY="none"
export L2_CACHE_MEMORY_BUDGET_MB=8192
export KEY_VALUE_STORE="block_based"
export BLOCK_BASED_PATH="/NVME0/guoran/rocks"
rm /NVME0/guoran/rocks/0-4/*
rm /NVME0/guoran/rocks/1-4/*
rm /NVME0/guoran/rocks/2-4/*
rm /NVME0/guoran/rocks/3-4/*
#export LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc.so.4:$LD_PRELOAD
export GRADIENT_SHUFFLE_USE_FP16=1

#/usr/local/cuda-11.4/bin/nsys profile --stat=true \
    python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    train.py \
    --dataset_format 'ofrecord' \
    --interaction_type dot \
    --embedding_type OneEmbedding \
    --bottom_mlp 512,256,$emb_size \
    --top_mlp 1024,1024,512,256 \
    --embedding_vec_size $emb_size \
    --learning_rate 24 \
    --batch_size $BATHSIZE \
    --data_dir $DATA_DIR \
    --loss_print_every_n_iter 10 \
    --eval_interval 1000 \
    --max_iter 10000 \
    --vocab_size $EMBD_SIZE \
    --data_part_num 5888 \
    --data_part_name_suffix_length 5 \
    --execution_mode 'graph' \
    --test_name 'train_graph_conisitent_'$DEVICE_NUM_PER_NODE'gpu'
    #--model_load_dir /tank/model_zoo/dlrm_baseline_params_emb$emb_size \
    # --dataset_format torch \
    # --model_load_dir /tank/xiexuan/dlrm/initial_parameters \
