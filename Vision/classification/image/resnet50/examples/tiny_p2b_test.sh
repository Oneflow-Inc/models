# set -aux
clear

DEVICE_NUM_PER_NODE=4
export ONEFLOW_TEST_DEVICE_NUM=4
MASTER_ADDR=127.0.0.1
NUM_NODES=1
NODE_RANK=0

export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED
export NCCL_LAUNCH_MODE=PARALLEL
echo NCCL_LAUNCH_MODE=$NCCL_LAUNCH_MODE
# export NCCL_DEBUG=INFO
export ONEFLOW_DEBUG_MODE=1
export ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE=1
export ONEFLOW_ENABLE_OFCCL=1

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
    test_p2b.py