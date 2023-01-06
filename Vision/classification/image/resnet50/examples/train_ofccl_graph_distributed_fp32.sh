# set -aux
clear

MASTER_ADDR=127.0.0.1
NUM_NODES=1
NODE_RANK=0
export GLOG_logtostderr=1
export ONEFLOW_ACTOR_ENABLE_LIGHT_ACTOR=0 # 禁用lightweight actor

export NCCL_PROTO=Simple
export NCCL_ALGO=Ring
# export NCCL_MAX_NCHANNELS=1
# export NCCL_MIN_NCHANNELS=1
# export NCCL_NTHREADS=64

export ONEFLOW_ENABLE_OFCCL=1
export ONEFLOW_OFCCL_SKIP_NEGO=1
export ONEFLOW_OFCCL_ORDERED_ISSUE_AR=1

export DEVICE_NUM_PER_NODE=2
if [ $DEVICE_NUM_PER_NODE = 4 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,4,5
fi

if [ -z $RUN_TYPE ];then
    RUN_TYPE="PURE"
    # RUN_TYPE="GDB"
    # RUN_TYPE="NSYS"
fi

if [ $ONEFLOW_ENABLE_OFCCL == "1" ]; then
    NSYS_FILE="ofccl_resnet"_${HOST}_${DEVICE_NUM_PER_NODE}_card_reverse_ordered_issue
else
    NSYS_FILE="nccl_resnet"_${HOST}_${DEVICE_NUM_PER_NODE}_card
fi

export PRINT_INTERVAL=2

export GLOG_vmodule=plan_util*=1,of_collective_actor*=1,of_collective_boxing_kernels*=1,collective_backend_ofccl*=1,hierarchical_sub_task_graph_builder_impl*=1,of_request_store*=1
# nn_graph*=1,
# export GLOG_v=1

export SHOW_ALL_PREPARED_COLL=1

export TRAVERSE_TIMES=10
export TOLERANT_UNPROGRESSED_CNT=10000
export BASE_CTX_SWITCH_THRESHOLD=80
export BOUNS_SWITCH_4_PROCESSED_COLL=0
export DEV_TRY_ROUND=10

# export ENABLE_VQ=1
# export TOLERANT_FAIL_CHECK_SQ_CNT=5000
# export CNT_BEFORE_QUIT=5

echo TRAVERSE_TIMES=$TRAVERSE_TIMES
echo TOLERANT_UNPROGRESSED_CNT=$TOLERANT_UNPROGRESSED_CNT
echo BASE_CTX_SWITCH_THRESHOLD=$BASE_CTX_SWITCH_THRESHOLD
echo BOUNS_SWITCH_4_PROCESSED_COLL=$BOUNS_SWITCH_4_PROCESSED_COLL
echo DEV_TRY_ROUND=$DEV_TRY_ROUND

if [ ! -z $BINARY ];then
    echo TOLERANT_FAIL_CHECK_SQ_CNT=$TOLERANT_FAIL_CHECK_SQ_CNT
    echo CNT_BEFORE_QUIT=$CNT_BEFORE_QUIT
fi

echo ONEFLOW_ENABLE_OFCCL=$ONEFLOW_ENABLE_OFCCL
echo ONEFLOW_OFCCL_SKIP_NEGO=$ONEFLOW_OFCCL_SKIP_NEGO
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
    export OFRECORD_PATH=/minio/sdd/dataset/imagenette/ofrecord
elif [ $HOST == "oneflow-16" ]; then
    export OFRECORD_PATH=/dataset/ImageNet/ofrecord
elif [ $HOST == "oneflow-25" ]; then
    export OFRECORD_PATH=/data/dataset/ImageNet/ofrecord
elif [ $HOST == "oneflow-26" ]; then
    export OFRECORD_PATH=/ssd/dataset/ImageNet/ofrecord
elif [ $HOST == "oneflow-27" ]; then
    export OFRECORD_PATH=/ssd/dataset/ImageNet/ofrecord
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

rm -rf ./log
mkdir ./log

if [ "$RUN_TYPE" == "PURE" ];then
    cmd="python3 -m oneflow.distributed.launch"
    export RESNET_ITER_FACTOR=40
elif [ "$RUN_TYPE" == "GDB" ];then
    cmd="gdb -ex r --args python3 -m oneflow.distributed.launch"
    export RESNET_ITER_FACTOR=40
elif [ "$RUN_TYPE" == "NSYS" ];then
    if [ ! -d "/home/panlichen/work/oneflow/log/nsys" ];then
        mkdir -p /home/panlichen/work/oneflow/log/nsys
    fi
    # cmd="nsys profile -f true --trace=cuda,cudnn,cublas,osrt,nvtx -o /home/panlichen/work/oneflow/log/nsys/$NSYS_FILE python3 -m oneflow.distributed.launch"
    cmd="nsys profile -f true -o /home/panlichen/work/oneflow/log/nsys/$NSYS_FILE python3 -m oneflow.distributed.launch"
    export RESNET_ITER_FACTOR=400
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
