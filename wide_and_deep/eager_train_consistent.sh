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
    eager_train_consistent.py \
        --model_save_dir ./checkpoints \
        --print_interval 100 \
        --deep_dropout_rate 0 \
        --eval_batchs 0 \
        --max_iter 200

