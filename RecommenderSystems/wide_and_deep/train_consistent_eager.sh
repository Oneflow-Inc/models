DEVICE_NUM_PER_NODE=2
MASTER_ADDR=127.0.0.1
NUM_NODES=1
NODE_RANK=0
DATA_DIR=/dataset/wdl_ofrecord/ofrecord
EMBD_SIZE=1603616

python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    train.py \
    --learning_rate 0.0005 \
    --batch_size 32 \
    --data_dir $DATA_DIR \
    --print_interval 100 \
    --deep_dropout_rate 0 \
    --max_iter 1100 \
    --hidden_units_num 7 \
    --hidden_size 1024 \
    --wide_vocab_size $EMBD_SIZE \
    --deep_vocab_size $EMBD_SIZE \
    --data_part_num 2 \
    --gpu_num_per_nod 1 \
    --node_ips $MASTER_ADDR \
    --val_batch_size 0 \
    --model_load_dir './initial_checkpoint' \
    --execution_mode 'eager' \
    --test_name 'train_eager_conisitent'
