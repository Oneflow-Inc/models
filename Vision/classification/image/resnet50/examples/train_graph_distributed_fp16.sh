# set -aux

DEVICE_NUM_PER_NODE=1
MASTER_ADDR=127.0.0.1
NUM_NODES=1
NODE_RANK=0

export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED
export NCCL_LAUNCH_MODE=PARALLEL
echo NCCL_LAUNCH_MODE=$NCCL_LAUNCH_MODE
export NCCL_DEBUG=INFO
export ONEFLOW_DEBUG_MODE=True
export CUDA_VISIABLE_DEVICES=2

CHECKPOINT_SAVE_PATH="./graph_distributed_fp16_checkpoints"
if [ ! -d "$CHECKPOINT_SAVE_PATH" ]; then
    mkdir $CHECKPOINT_SAVE_PATH
fi

#OFRECORD_PATH=PATH_TO_IMAGENET_OFRECORD
OFRECORD_PATH="/dataset/imagenet/ofrecord/"

OFRECORD_PART_NUM=256
LEARNING_RATE=1.536
MOM=0.875
EPOCH=50
#TRAIN_BATCH_SIZE=19
#TRAIN_BATCH_SIZE=192
TRAIN_BATCH_SIZE=80
VAL_BATCH_SIZE=5
#VAL_BATCH_SIZE=50

# SRC_DIR=/path/to/models/resnet50
SRC_DIR=$(realpath $(dirname $0)/..)

python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    $SRC_DIR/train.py \
        --save $CHECKPOINT_SAVE_PATH \
        --ofrecord-path $OFRECORD_PATH \
        --ofrecord-part-num $OFRECORD_PART_NUM \
        --num-devices-per-node $DEVICE_NUM_PER_NODE \
        --lr $LEARNING_RATE \
        --momentum $MOM \
        --num-epochs $EPOCH \
        --train-batch-size $TRAIN_BATCH_SIZE \
        --val-batch-size $VAL_BATCH_SIZE \
        --graph \
        --use-fp16 \
        --metric-local True \
        --metric-train-acc True \
        --fuse-bn-relu \
        --fuse-bn-add-relu \
        --use-gpu-decode \
        --channel-last \
