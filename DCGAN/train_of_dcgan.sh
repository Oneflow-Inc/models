  
set -aux

BASE_PATH='./'
LEARNING_RATE=le-4
EPOCH=100
TRAIN_BATCH_SIZE=200
# LOAD_CHECKPOINT=

python3 of_dcgan.py \
    -lr $LEARNING_RATE \
    -e $EPOCH \
    --batch_size $TRAIN_BATCH_SIZE \
    --val_batch_size $VAL_BATCH_SIZE \
    --path $BASE_PATH \
    # --load $LOAD_CHECKPOINT
