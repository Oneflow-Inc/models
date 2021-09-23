DEVICE_NUM_PER_NODE=2
MASTER_ADDR=127.0.0.1
NUM_NODES=1
NODE_RANK=0

# export CUDA_VISIBLE_DEVICES=3
python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    train.py \
        --deep_vocab_size 1603616 \
        --wide_vocab_size 1603616 \
        --deep_embedding_vec_size 16 \
        --hidden_units_num 7 \
        --hidden_size 1024 \
        --deep_dropout_rate 0 \
        --print_interval 100 \
        --max_iter 1000 \
        --ddp \
        --execution_mode 'eager'\
        --save_initial_model \
        --model_save_dir "./checkpoints"

