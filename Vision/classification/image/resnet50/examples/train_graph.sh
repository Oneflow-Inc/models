# set -aux

export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED

CHECKPOINT_SAVE_PATH="./graph_checkpoints"
if [ ! -d "$CHECKPOINT_SAVE_PATH" ]; then
    mkdir $CHECKPOINT_SAVE_PATH
fi

OFRECORD_PATH="./mini-imagenet/ofrecord"

if [ ! -d "$OFRECORD_PATH" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/dataset/imagenet/mini-imagenet.zip
    unzip mini-imagenet.zip
fi

OFRECORD_PART_NUM=1
LEARNING_RATE=0.256
MOM=0.875
EPOCH=90
TRAIN_BATCH_SIZE=50
VAL_BATCH_SIZE=50

# SRC_DIR=/path/to/models/resnet50
SRC_DIR=$(realpath $(dirname $0)/..)

python3 $SRC_DIR/train.py \
    --ofrecord-path $OFRECORD_PATH \
    --ofrecord-part-num $OFRECORD_PART_NUM \
    --num-devices-per-node 1 \
    --lr $LEARNING_RATE \
    --momentum $MOM \
    --num-epochs $EPOCH \
    --warmup-epochs 0 \
    --train-batch-size $TRAIN_BATCH_SIZE \
    --val-batch-size $VAL_BATCH_SIZE \
    --save $CHECKPOINT_SAVE_PATH \
    --samples-per-epoch 50 \
    --val-samples-per-epoch 50 \
    --use-gpu-decode \
    --scale-grad \
    --graph \
