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
    eval.py \
        --deep_vocab_size 2322444\
        --wide_vocab_size 2322444\
        --hidden_units_num 2\
        --deep_embedding_vec_size 16\
        --batch_size 16384\
        --model_save_dir ./checkpoints \
        --print_interval 100 \
        --deep_dropout_rate 0 \
        --max_iter 300 \
        --execution_mode 'eager' \
        --eval_name 'consistent_eager'

