NUM_NODES=$1
DEVICE_NUM_PER_NODE=$2
BATHSIZE=$3
EMBD_SIZE=$4
DEEP_VEC_SIZE=$5
PREFIX=$6
SUFFIX=$7
MASTER_ADDR=127.0.0.1
NODE_RANK=0
DATA_DIR=/dataset/f9f659c5/wdl_ofrecord



python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
  train.py \
    --learning_rate 0.001 \
    --batch_size $BATHSIZE \
    --data_dir $DATA_DIR \
    --loss_print_every_n_iter 10 \
    --eval_interval 10 \
    --dropout_rate 0.5 \
    --max_iter 110 \
    --vocab_size $EMBD_SIZE \
    --embedding_vec_size $DEEP_VEC_SIZE \
    --data_part_num 256 \
    --data_part_name_suffix_length 5 \
    --execution_mode 'graph' \
    --test_name 'train_eager_graph_'$DEVICE_NUM_PER_NODE'gpu' > $PREFIX'_n'$NUM_NODES'_g'$DEVICE_NUM_PER_NODE'_b'$BATHSIZE'_h'$HIDDEN_UNITS_NUM'_log_'$PREFIX

