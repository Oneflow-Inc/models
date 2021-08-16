# set -aux

MASTER_ADDR=127.0.0.1
MASTER_PORT=17788
TOTAL_DEVICE_NUM=4
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
LEARNING_RATE=0.768
MOM=0.875
EPOCH=2000
TRAIN_BATCH_SIZE_PER_DEVICE=64
VAL_BATCH_SIZE=50

NCCL_DEBUG=INFO \
python3 -m oneflow.distributed.launch \
    --nproc_per_node $TOTAL_DEVICE_NUM \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    eager/train_ddp.py \
    --save_checkpoint_path $CHECKPOINT_PATH \
    --ofrecord_path $OFRECORD_PATH \
    --ofrecord_part_num $OFRECORD_PART_NUM \
    --learning_rate $LEARNING_RATE \
    --mom $MOM \
    --epochs $EPOCH \
    --train_batch_size $TRAIN_BATCH_SIZE_PER_DEVICE \
    --val_batch_size $VAL_BATCH_SIZE
