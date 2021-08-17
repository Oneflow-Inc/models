set -aux

OFRECORD_PATH="ofrecord"
if [ ! -d "$OFRECORD_PATH" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenette_ofrecord.tar.gz
    tar zxf imagenette_ofrecord.tar.gz
fi

CHECKPOINT_PATH="checkpoints"
if [ ! -d "$CHECKPOINT_PATH" ]; then
    mkdir $CHECKPOINT_PATH
fi

LEARNING_RATE=0.001
MOM=0.9
EPOCH=20
TRAIN_BATCH_SIZE=16
VAL_BATCH_SIZE=16
NUM_CLASSES=1000
LABEL_SMOOTH_RATE=0.1
#LOAD_CHECKPOINT="path/to/your_pretrain_model" # LOAD PREVIOUS CHECKPOINT 

python3 graph/train_labelsmooth.py \
    --save_checkpoint_path $CHECKPOINT_PATH \
    --ofrecord_path $OFRECORD_PATH \
    --learning_rate $LEARNING_RATE \
    --mom $MOM \
    --epochs $EPOCH \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --val_batch_size $VAL_BATCH_SIZE \
    --num_classes $NUM_CLASSES \
    --label_smooth_rate $LABEL_SMOOTH_RATE
    #--load_checkpoint $LOAD_CHECKPOINT