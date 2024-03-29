export PYTHONPATH=$PWD:$PYTHONPATH
set -aux

OFRECORD_PATH="CamVid"
if [ ! -d "$OFRECORD_PATH" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/seg/FODDET/CamVid.zip
    unzip CamVid.zip
fi


LEARNING_RATE=0.001
EPOCH=50
TRAIN_BATCH_SIZE=8
DATA_DIR="./CamVid"
SAVE_PATH="./checkpoint"


python3 train/train.py --data_dir $DATA_DIR \
                       --save_checkpoint_path $SAVE_PATH \
                       --learning_rate $LEARNING_RATE \
                       --epochs $EPOCH \
                       --train_batch_size $TRAIN_BATCH_SIZE \
