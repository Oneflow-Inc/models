# set -aux
clear

export ONEFLOW_ENABLE_OFCCL=1
export ONEFLOW_OFCCL_SKIP_NEGO=1
export ONEFLOW_OFCCL_DUMMY_KERNEL=0

export GLOG_logtostderr=1

# export RUN_TYPE=GDB

export NCCL_PROTO=Simple
export NCCL_ALGO=Ring
#export NCCL_MAX_NCHANNELS=2
#export NCCL_MIN_NCHANNELS=2
# export NCCL_NTHREADS=64


export ONEFLOW_ACTOR_ENABLE_LIGHT_ACTOR=0 # 禁用lightweight actor

# if [ -z $DEVICE_NUM_PER_NODE ];then
#     DEVICE_NUM_PER_NODE=4
# fi
# export CUDA_VISIBLE_DEVICES=0,1,4,5

if [ -z $DEVICE_NUM_PER_NODE ];then
    DEVICE_NUM_PER_NODE=8
fi

MASTER_ADDR=127.0.0.1
NUM_NODES=1
NODE_RANK=0

export GLOG_vmodule=nn_graph*=1,plan_util*=1,of_collective_actor*=1,of_collective_boxing_kernels*=1,collective_backend_ofccl*=1,hierarchical_sub_task_graph_builder_impl=1
# export GLOG_v=1

echo ONEFLOW_ENABLE_OFCCL=$ONEFLOW_ENABLE_OFCCL
echo ONEFLOW_OFCCL_SKIP_NEGO=$ONEFLOW_OFCCL_SKIP_NEGO
echo ONEFLOW_OFCCL_DUMMY_KERNEL=$ONEFLOW_OFCCL_DUMMY_KERNEL
echo NCCL_PROTO=$NCCL_PROTO
echo NCCL_ALGO=$NCCL_ALGO
echo NCCL_MAX_NCHANNELS=$NCCL_MAX_NCHANNELS
echo NCCL_NTHREADS=$NCCL_NTHREADS
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

# export NCCL_MAX_NCHANNELS=1
# export NCCL_NTHREADS=64

CHECKPOINT_SAVE_PATH="./graph_distributed_fp32_checkpoints"
if [ ! -d "$CHECKPOINT_SAVE_PATH" ]; then
    mkdir $CHECKPOINT_SAVE_PATH
fi

if [ $HOST == "oneflow-15" ]; then
    export OFRECORD_PATH=/home/panlichen/dataset/ImageNet/ofrecord
elif [ $HOST == "oneflow-16" ]; then
    export OFRECORD_PATH=/dataset/ImageNet/ofrecord
elif [ $HOST == "oneflow-25" ]; then
    export OFRECORD_PATH=/data/dataset/ImageNet/ofrecord
elif [ $HOST == "oneflow-26" ]; then
    export OFRECORD_PATH=/data/home/panlichen/ImageNet/ofrecord
elif [ $HOST == "oneflow-28" ]; then
    export OFRECORD_PATH=/ssd/dataset/ImageNet/ofrecord
else
    echo "NO LEGAL HOST, exit."
    exit 1
fi

OFRECORD_PART_NUM=256
LEARNING_RATE=0.768
MOM=0.875
EPOCH=50
# TRAIN_BATCH_SIZE=96
# VAL_BATCH_SIZE=50
TRAIN_BATCH_SIZE=20
VAL_BATCH_SIZE=20
# TRAIN_BATCH_SIZE=50
# VAL_BATCH_SIZE=50

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

echo cmd=$cmd

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
        # > /home/panlichen/work/oneflow/log/oneflow.log 2>&1
