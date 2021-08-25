# set -aux
rm core.*

DEVICE_NUM_PER_NODE=1
MASTER_ADDR=127.0.0.1
NUM_NODES=1
NODE_RANK=0

export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED
export NCCL_LAUNCH_MODE=PARALLEL
echo NCCL_LAUNCH_MODE=$NCCL_LAUNCH_MODE
export NCCL_DEBUG=INFO 

OFRECORD_PATH="/dataset/ImageNet/ofrecord/"
# if [ ! -d "$OFRECORD_PATH" ]; then
#     wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenette_ofrecord.tar.gz
#     tar zxf imagenette_ofrecord.tar.gz
# fi

CHECKPOINT_PATH="eager_checkpoints"
if [ ! -d "$CHECKPOINT_PATH" ]; then
    mkdir $CHECKPOINT_PATH
fi
# LOAD_CHECKPOINT="graph_amp_checkpoints_lr_1.024_batch_128_warmup_5/epoch_8_val_acc_0.603240"

OFRECORD_PART_NUM=256
LEARNING_RATE=0.1
# LEARNING_RATE=0.256
# MOM=0.875
MOM=0.0
EPOCH=50
# TRAIN_BATCH_SIZE_PER_DEVICE=192
TRAIN_BATCH_SIZE_PER_DEVICE=64
VAL_BATCH_SIZE_PER_DEVICE=50

export ONEFLOW_DEBUG_MODE=1
python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    eager/train.py \
    --loss_print_every_n_iter=1 \
    --load_checkpoint 'init_ckpt' \
    --save_checkpoint_path $CHECKPOINT_PATH \
    --ofrecord_path $OFRECORD_PATH \
    --ofrecord_part_num $OFRECORD_PART_NUM \
    --process_num_per_node $DEVICE_NUM_PER_NODE \
    --learning_rate $LEARNING_RATE \
    --momentum $MOM \
    --num_epochs $EPOCH \
    --train_batch_size_per_device $TRAIN_BATCH_SIZE_PER_DEVICE \
    --val_batch_size_per_device $VAL_BATCH_SIZE_PER_DEVICE \
    --label_smoothing=0.0 \
    --warmup_epochs 5
    # --load_checkpoint $LOAD_CHECKPOINT \
    # --use_fp16 \

