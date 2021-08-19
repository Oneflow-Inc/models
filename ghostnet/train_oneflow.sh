set -aux

OFRECORD_PATH="/mnt/local/fengyuchao/Data/ofrecord"
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
EPOCH=1000
TRAIN_BATCH_SIZE=16
VAL_BATCH_SIZE=16
MODEL=ghostnet

# LOAD PREVIOUS CHECKPOINT 
# LOAD_CHECKPOINT=$CHECKPOINT_PATH/epoch_484_val_acc_0.908418

python3 train_oneflow.py \
    --save_checkpoint_path $CHECKPOINT_PATH \
    --ofrecord_path $OFRECORD_PATH \
    --learning_rate $LEARNING_RATE \
    --mom $MOM \
    --epochs $EPOCH \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --val_batch_size $VAL_BATCH_SIZE \
    --model $MODEL \
    #--load_checkpoint $LOAD_CHECKPOINT


