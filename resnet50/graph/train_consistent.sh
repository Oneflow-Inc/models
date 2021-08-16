# set -aux

TOTAL_DEVICE_NUM=4
MASTER_ADDR=127.0.0.1
NUM_NODES=1
NODE_RANK=0

OFRECORD_PATH="/DATA/disk1/ImageNet/ofrecord/"
# if [ ! -d "$OFRECORD_PATH" ]; then
#     wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenette_ofrecord.tar.gz
#     tar zxf imagenette_ofrecord.tar.gz
# fi

CHECKPOINT_PATH="ddp_checkpoints"
if [ ! -d "$CHECKPOINT_PATH" ]; then
    mkdir $CHECKPOINT_PATH
fi

OFRECORD_PART_NUM=256
# LEARNING_RATE=0.768
LEARNING_RATE=0.384
MOM=0.875
EPOCH=2000
TRAIN_BATCH_SIZE_PER_DEVICE=80
VAL_BATCH_SIZE=50

NCCL_DEBUG=INFO python3 -m oneflow.distributed.launch \
    --nproc_per_node $TOTAL_DEVICE_NUM \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    graph/train_consistent.py \
    --save_checkpoint_path $CHECKPOINT_PATH \
    --ofrecord_path $OFRECORD_PATH \
    --ofrecord_part_num $OFRECORD_PART_NUM \
    --device_num $TOTAL_DEVICE_NUM \
    --learning_rate $LEARNING_RATE \
    --mom $MOM \
    --epochs $EPOCH \
    --train_batch_size $TRAIN_BATCH_SIZE_PER_DEVICE \
    --val_batch_size $VAL_BATCH_SIZE
