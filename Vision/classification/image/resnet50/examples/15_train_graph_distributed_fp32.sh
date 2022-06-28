# set -aux
clear

if [ -z $DEVICE_NUM_PER_NODE ];then
    DEVICE_NUM_PER_NODE=2
fi
MASTER_ADDR=127.0.0.1
NUM_NODES=1
NODE_RANK=0

GLOG_vmodule="nn_graph=1, plan_util=1, of_collective_actor=1, of_collective_boxing_kernels=1"
GLOG_v=0
GLOG_logtostderr=1

echo ONEFLOW_OFCCL_SKIP_NEGO=$ONEFLOW_OFCCL_SKIP_NEGO
echo ONEFLOW_OFCCL_CHAIN=$ONEFLOW_OFCCL_CHAIN
echo GLOG_vmodule=$GLOG_vmodule
echo GLOG_v=$GLOG_v
echo GLOG_logtostderr=$GLOG_logtostderr

echo DEVICE_NUM_PER_NODE=$DEVICE_NUM_PER_NODE

export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED
export NCCL_LAUNCH_MODE=PARALLEL
echo NCCL_LAUNCH_MODE=$NCCL_LAUNCH_MODE
# export NCCL_DEBUG=INFO
export ONEFLOW_DEBUG_MODE=1
export ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE=1
export ONEFLOW_ENABLE_OFCCL=1

CHECKPOINT_SAVE_PATH="./graph_distributed_fp32_checkpoints"
if [ ! -d "$CHECKPOINT_SAVE_PATH" ]; then
    mkdir $CHECKPOINT_SAVE_PATH
fi

OFRECORD_PATH=/dataset/ImageNet/ofrecord

OFRECORD_PART_NUM=256
LEARNING_RATE=0.768
MOM=0.875
EPOCH=50
# TRAIN_BATCH_SIZE=96
# VAL_BATCH_SIZE=50
TRAIN_BATCH_SIZE=20
VAL_BATCH_SIZE=20

# SRC_DIR=/path/to/models/resnet50
SRC_DIR=$(realpath $(dirname $0)/..)

if [ $ONEFLOW_ENABLE_OFCCL == "1" ]; then
    NSYS_FILE="ofccl_resnet"
else
    NSYS_FILE="nccl_resnet"
fi

rm -rf ./log
mkdir ./log

if [ -z $RUN_TYPE ];then
    RUN_TYPE="PURE"
fi

if [ "$RUN_TYPE" == "PURE" ];then
    cmd="python3 -m oneflow.distributed.launch"
elif [ "$RUN_TYPE" == "GDB" ];then
    cmd="gdb -ex r --args python3 -m oneflow.distributed.launch"
elif [ "$RUN_TYPE" == "NSYS" ];then
    cmd="nsys profile -f true --trace=cuda,cudnn,cublas,osrt,nvtx -o nsys/$NSYS_FILE python3 -m oneflow.distributed.launch"
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

$cmd \
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
        --use-gpu-decode \
        --scale-grad \
        --graph \
        --fuse-bn-relu \
        --fuse-bn-add-relu \
