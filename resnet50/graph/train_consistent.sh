# set -aux

TOTAL_DEVICE_NUM=1
MASTER_ADDR=127.0.0.1
NUM_NODES=1
NODE_RANK=0

OFRECORD_PATH="ofrecord"
if [ ! -d "$OFRECORD_PATH" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenette_ofrecord.tar.gz
    tar zxf imagenette_ofrecord.tar.gz
fi

CHECKPOINT_PATH="ddp_checkpoints"
if [ ! -d "$CHECKPOINT_PATH" ]; then
    mkdir $CHECKPOINT_PATH
fi

OFRECORD_PART_NUM=1
LEARNING_RATE=0.001
MOM=0.9
EPOCH=20
TRAIN_BATCH_SIZE_PER_DEVICE=16
VAL_BATCH_SIZE=16

python3 -m oneflow.distributed.launch \
    --nproc_per_node $TOTAL_DEVICE_NUM \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    graph/train_consistent.py \
    --save_checkpoint_path $CHECKPOINT_PATH \
    --ofrecord_path $OFRECORD_PATH \
    --learning_rate $LEARNING_RATE \
    --mom $MOM \
    --epochs $EPOCH \
    --train_batch_size $TRAIN_BATCH_SIZE_PER_DEVICE \
    --val_batch_size $VAL_BATCH_SIZE
