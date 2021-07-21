set -aux

IMDB_PATH="aclImdb"
if [ ! -d "$IMDB_PATH" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/aclImdb_v1.tar.gz
    tar zxf aclImdb_v1.tar.gz
fi




LEARNING_RATE=0.001
EPOCH=15
TRAIN_BATCH_SIZE=16
VAL_BATCH_SIZE=16
CHECKPOINT_PATH="checkpoints"

python3 train_textcnn.py \
    --save_checkpoint_path $CHECKPOINT_PATH \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCH \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --val_batch_size $VAL_BATCH_SIZE \
    --kernel_size 3 4 5