set -aux

DATA_PATH="imagenette2"
if [ ! -d "$DATA_PATH" ]; then
    wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
    tar zxf imagenette2.tgz
fi

CHECKPOINT_PATH="./torch_checkpoints"
if [ ! -d "$CHECKPOINT_PATH" ]; then
    mkdir $CHECKPOINT_PATH
fi

LEARNING_RATE=0.1
MOM=0.9
WD=0.0001
EPOCH=90
BATCH_SIZE=64

export CUDA_VISIBLE_DEVICES=2

python3 train_torch.py $DATA_PATH \
    -j 8 \
    --checkpoint-path $CHECKPOINT_PATH \
    --learning-rate $LEARNING_RATE \
    --momentum $MOM \
    --weight-decay $WD \
    --epochs $EPOCH \
    --batch-size $BATCH_SIZE \
    | tee ${CHECKPOINT_PATH}/log.txt
