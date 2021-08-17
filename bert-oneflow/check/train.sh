set -aux

OFRECORD_PATH="wiki_ofrecord_seq_len_128_example"
if [ ! -d "$OFRECORD_PATH" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/wiki_ofrecord_seq_len_128_example.tgz
    tar zxf wiki_ofrecord_seq_len_128_example.tgz
fi

CHECKPOINT_PATH="checkpoints"
if [ ! -d "$CHECKPOINT_PATH" ]; then
    mkdir $CHECKPOINT_PATH
fi

LEARNING_RATE=0.0001
EPOCH=20
TRAIN_BATCH_SIZE=16
VAL_BATCH_SIZE=16

python3 check/check.py \
    --ofrecord-path $OFRECORD_PATH \
    --lr $LEARNING_RATE \
    --epochs $EPOCH \
    --train-batch-size $TRAIN_BATCH_SIZE \
    --val-batch-size $VAL_BATCH_SIZE \
