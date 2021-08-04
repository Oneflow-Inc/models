set -aux

OFRECORD_PATH="ofrecord"
if [ ! -d "$OFRECORD_PATH" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenette_ofrecord.tar.gz
    tar zxf imagenette_ofrecord.tar.gz
fi

CHECKPOINT_PATH="./of_checkpoints"
if [ ! -d "$CHECKPOINT_PATH" ]; then
    mkdir $CHECKPOINT_PATH
fi

LEARNING_RATE=0.001
MOM=0.9
WD=0.0005
EPOCH=90
BATCH_SIZE=16

# LOAD PREVIOUS CHECKPOINT 
# LOAD_CHECKPOINT=$CHECKPOINT_PATH/epoch_2_val_acc_0.111168

export CUDA_VISIBLE_DEVICES=1
# export GLOG_v=2

python3 graph_train.py \
    --ofrecord-path $OFRECORD_PATH \
    --checkpoint-path $CHECKPOINT_PATH \
    --learning-rate $LEARNING_RATE \
    --momentum $MOM \
    --weight-decay $WD \
    --epochs $EPOCH \
    --batch $BATCH_SIZE \
    # | tee ${CHECKPOINT_PATH}/log.txt
    # --load_checkpoint $LOAD_CHECKPOINT
