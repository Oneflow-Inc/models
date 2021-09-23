# set -aux
# export PYTHONPATH=/home/luyang/Oneflow/oneflow/python:$PYTHONPATH
# nohup  bash examples/train_graph_nsight.sh  >  nsight_graph_resnet50_fp32_1n1g_bz96_20iter_gap1_210923@be983659d.log  2>&1 &
# nohup  bash examples/train_graph_nsight.sh  >  nsight_graph_resnet50_fp32_1n1g_bz96_20iter_gap1_profile_210923@be983659d.log  2>&1 &
DEVICE_NUM_PER_NODE=1
MASTER_ADDR=127.0.0.1
NUM_NODES=1
NODE_RANK=0

export ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE=True
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

OFRECORD_PATH="/dataset/ImageNet/ofrecord"

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

/home/luyang/nsight-systems-2021.2.1/bin/nsys profile --stats=true -o nsight_graph_resnet50_fp32_1n1g_bz96_20iter_gap20_profile_210923@be983659d.qdrep \
python3    $SRC_DIR/train_nsight_profile.py \
        --save $CHECKPOINT_SAVE_PATH \
        --ofrecord-path $OFRECORD_PATH \
        --ofrecord-part-num $OFRECORD_PART_NUM \
        --num-devices-per-node $DEVICE_NUM_PER_NODE \
        --lr $LEARNING_RATE \
        --momentum $MOM \
        --num-epochs $EPOCH \
        --train-batch-size $TRAIN_BATCH_SIZE \
        --val-batch-size $VAL_BATCH_SIZE \
        --scale-grad \
        --graph \
        --nsight-step 20 \
        --print-interval 20 \
        # --use-gpu-decode \

