  
set -aux

DATA_PATH="mnist"
if [ ! -d "$DATA_PATH" ]; then
    wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/Dataset/mnist.zip
    unzip mnist.zip
fi


BASE_PATH='./dcgan'
LEARNING_RATE=0.0001
EPOCH=1
BATCH_SIZE=256
SAVE=True
# LOAD_CHECKPOINT=

python3 graph/train_of_dcgan_graph.py \
    -lr $LEARNING_RATE \
    -e $EPOCH \
    --batch_size $BATCH_SIZE \
    --path $BASE_PATH \
    --save $SAVE \
    --data_dir $DATA_PATH \
    # --load $LOAD_CHECKPOINT
