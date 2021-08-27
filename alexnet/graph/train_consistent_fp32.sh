# set -aux

DEVICE_NUM_PER_NODE=8
MASTER_ADDR=127.0.0.1
NUM_NODES=1
NODE_RANK=0

export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED
export NCCL_LAUNCH_MODE=PARALLEL
echo NCCL_LAUNCH_MODE=$NCCL_LAUNCH_MODE
export NCCL_DEBUG=INFO 

OFRECORD_PATH="/datasets/imagenet/ofrecord/"
# if [ ! -d "$OFRECORD_PATH" ]; then
#     wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenette_ofrecord.tar.gz
#     tar zxf imagenette_ofrecord.tar.gz
# fi

CHECKPOINT_PATH="graph_checkpoints"
if [ ! -d "$CHECKPOINT_PATH" ]; then
    mkdir $CHECKPOINT_PATH
fi

OFRECORD_PART_NUM=256
LEARNING_RATE=0.01
MOM=0.9
EPOCH=90
TRAIN_BATCH_SIZE_PER_DEVICE=512
VAL_BATCH_SIZE_PER_DEVICE=512

python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    graph/train_consistent.py \
    --save_checkpoint_path $CHECKPOINT_PATH \
    --ofrecord_path $OFRECORD_PATH \
    --ofrecord_part_num $OFRECORD_PART_NUM \
    --process_num_per_node $DEVICE_NUM_PER_NODE \
    --learning_rate $LEARNING_RATE \
    --momentum $MOM \
    --num_epochs $EPOCH \
    --train_batch_size_per_device $TRAIN_BATCH_SIZE_PER_DEVICE \
    --val_batch_size_per_device $VAL_BATCH_SIZE_PER_DEVICE \
    --nccl_fusion_threshold_mb=16 \
    --nccl_fusion_max_ops=24