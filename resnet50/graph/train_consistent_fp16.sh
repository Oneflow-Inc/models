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

OFRECORD_PATH="/dataset/ImageNet/ofrecord/"
# if [ ! -d "$OFRECORD_PATH" ]; then
#     wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenette_ofrecord.tar.gz
#     tar zxf imagenette_ofrecord.tar.gz
# fi

CHECKPOINT_PATH="graph_amp_checkpoints"
if [ ! -d "$CHECKPOINT_PATH" ]; then
    mkdir $CHECKPOINT_PATH
fi

PRETRAIN_MODEL_PATH="resnet50_imagenet_pretrain_model"
if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/resnet50_imagenet_pretrain_model.tar.gz
  tar zxf resnet50_imagenet_pretrain_model.tar.gz
fi

# LOAD_CHECKPOINT="graph_amp_checkpoints_lr_1.024_batch_128_warmup_5/epoch_8_val_acc_0.603240"
LOAD_CHECKPOINT="/DATA/disk1/ldp/OneFlow-Benchmark/Classification/cnns/output/snapshots/model_save-20210824083821/snapshot_epoch_0_graph"

OFRECORD_PART_NUM=256
# LEARNING_RATE=1.536
LEARNING_RATE=1.024
MOM=0.875
# EPOCH=90
EPOCH=50
TRAIN_BATCH_SIZE_PER_DEVICE=192
# TRAIN_BATCH_SIZE_PER_DEVICE=128
VAL_BATCH_SIZE_PER_DEVICE=50

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
        --label_smoothing=0.1 \
        --warmup_epochs 5 \
        --use_fp16 \
        --nccl_fusion_threshold_mb=16 \
        --nccl_fusion_max_ops=24 \
        # --load_checkpoint $LOAD_CHECKPOINT \
