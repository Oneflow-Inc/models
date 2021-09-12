# set -aux

DEVICE_NUM_PER_NODE=2
MASTER_ADDR=127.0.0.1
NUM_NODES=1
NODE_RANK=0

export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED
export NCCL_LAUNCH_MODE=PARALLEL
echo NCCL_LAUNCH_MODE=$NCCL_LAUNCH_MODE
# export NCCL_DEBUG=INFO
# export ONEFLOW_DEBUG_MODE=True

CHECKPOINT_SAVE_PATH="./ddp_checkpoints_simple3"
if [ ! -d "$CHECKPOINT_SAVE_PATH" ]; then
    mkdir $CHECKPOINT_SAVE_PATH
fi

CHECKPOINT_LOAD_PATH="./init_ckpt_by_lazy"
if [ ! -d "$CHECKPOINT_LOAD_PATH" ]; then
    wget http://oneflow-static.oss-cn-beijing.aliyuncs.com/resnet50_init_ckpt_by_lazy.tgz
    tar zxf resnet50_init_ckpt_by_lazy.tgz
    rm -rf $CHECKPOINT_LOAD_PATH/System-Train-TrainStep-TrainNet
fi

OFRECORD_PATH=/dataset/ImageNet/ofrecord/
OFRECORD_PART_NUM=256
LEARNING_RATE=0.128
MOM=0.875
EPOCH=50
TRAIN_BATCH_SIZE=16
VAL_BATCH_SIZE=1

# SRC_DIR=/path/to/models/resnet50
SRC_DIR=$(realpath $(dirname $0)/..)

python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --logdir logdist \
    $SRC_DIR/train.py \
        --save $CHECKPOINT_SAVE_PATH \
        --load $CHECKPOINT_LOAD_PATH \
        --warmup-epochs 0 \
        --print-interval 1  \
        --ofrecord-path $OFRECORD_PATH \
        --ofrecord-part-num $OFRECORD_PART_NUM \
        --num-devices-per-node $DEVICE_NUM_PER_NODE \
        --lr $LEARNING_RATE \
        --momentum $MOM \
        --num-epochs $EPOCH \
        --total-batches 200 \
        --train-batch-size $TRAIN_BATCH_SIZE \
        --val-batch-size $VAL_BATCH_SIZE \
        --ddp \
