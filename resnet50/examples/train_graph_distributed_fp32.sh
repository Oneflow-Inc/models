# set -aux
# export PYTHONPATH=/home/luyang/Oneflow/oneflow/python:$PYTHONPATH
# nohup  bash examples/train_graph_distributed_fp32.sh  >  graph_resnet50_fp32_1n8g_bz96@fdeb09426.log  2>&1 &
DEVICE_NUM_PER_NODE=1
MASTER_ADDR=127.0.0.1
NUM_NODES=1
NODE_RANK=0

export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED
export NCCL_LAUNCH_MODE=PARALLEL
echo NCCL_LAUNCH_MODE=$NCCL_LAUNCH_MODE
# export NCCL_DEBUG=INFO
# export ONEFLOW_DEBUG_MODE=True

CHECKPOINT_SAVE_PATH="./graph_distributed_fp32_checkpoints"
if [ ! -d "$CHECKPOINT_SAVE_PATH" ]; then
    mkdir $CHECKPOINT_SAVE_PATH
fi

OFRECORD_PATH="/datasets/imagenet/ofrecord"

OFRECORD_PART_NUM=256
LEARNING_RATE=0.096
MOM=0.875
EPOCH=50
TRAIN_BATCH_SIZE=96
VAL_BATCH_SIZE=50

# SRC_DIR=/path/to/models/resnet50
SRC_DIR=$(realpath $(dirname $0)/..)

# python3 -m oneflow.distributed.launch \
#     --nproc_per_node $DEVICE_NUM_PER_NODE \
#     --nnodes $NUM_NODES \
#     --node_rank $NODE_RANK \
#     --master_addr $MASTER_ADDR \
python3    $SRC_DIR/train.py \
        --save $CHECKPOINT_SAVE_PATH \
        --ofrecord-path $OFRECORD_PATH \
        --ofrecord-part-num $OFRECORD_PART_NUM \
        --num-devices-per-node $DEVICE_NUM_PER_NODE \
        --lr $LEARNING_RATE \
        --momentum $MOM \
        --num-epochs $EPOCH \
        --train-batch-size $TRAIN_BATCH_SIZE \
        --val-batch-size $VAL_BATCH_SIZE \
        --use-gpu-decode \
        --scale-grad \
        --graph
