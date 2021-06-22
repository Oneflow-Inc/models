set -aux

BASE_PATH='./'
LEARNING_RATE=2e-4
EPOCH=200
TRAIN_BATCH_SIZE=32
# LOAD_CHECKPOINT=

python3 ./train_oneflow.py \
    -lr $LEARNING_RATE \
    -e $EPOCH \
    --batch_size $TRAIN_BATCH_SIZE \
    --path $BASE_PATH #\
    # --load $LOAD_CHECKPOINT