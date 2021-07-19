set -aux

OFRECORD_PATH="./ofrecord/hustcolor"

CHECKPOINT_PATH="checkpoints"
if [ ! -d "$CHECKPOINT_PATH" ]; then
    mkdir $CHECKPOINT_PATH
fi

LEARNING_RATE=0.001
MOM=0.9
EPOCH=1000
TRAIN_BATCH_SIZE=16
VAL_BATCH_SIZE=16

python train_oneflow.py \
    --save_checkpoint_path $CHECKPOINT_PATH \
    --ofrecord_path $OFRECORD_PATH \
    --learning_rate $LEARNING_RATE \
    --mom $MOM \
    --epochs $EPOCH \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --val_batch_size $VAL_BATCH_SIZE \
    # --load_checkpoint $LOAD_CHECKPOINT


