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

LEARNING_RATE=0.01
MOM=0.9
WD=0.0005
EPOCH=90
TRAIN_BATCH_SIZE=128
VAL_BATCH_SIZE=256

# LOAD PREVIOUS CHECKPOINT 
# LOAD_CHECKPOINT=$CHECKPOINT_PATH/epoch_2_val_acc_0.111168

export CUDA_VISIBLE_DEVICES=1
# export GLOG_v=2

python3 graph_train.py \
    --save_checkpoint_path $CHECKPOINT_PATH \
    --ofrecord_path $OFRECORD_PATH \
    --learning_rate $LEARNING_RATE \
    --mom $MOM \
    --wd $WD \
    --epochs $EPOCH \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --val_batch_size $VAL_BATCH_SIZE \
    # --load_checkpoint $LOAD_CHECKPOINT


