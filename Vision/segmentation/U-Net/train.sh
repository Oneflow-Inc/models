set -aux



OFRECORD_PATH="test_image"
if [ ! -d "$OFRECORD_PATH" ]; then
    wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/Dataset/Seg/ISBI_Challenge_cell_segmentation/test_image.tar.gz
    tar xvf test_image.tar.gz
fi

OFRECORD_PATH="train_image"
if [ ! -d "$OFRECORD_PATH" ]; then
    wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/Dataset/Seg/ISBI_Challenge_cell_segmentation/train_image.tar.gz
    tar xvf train_image.tar.gz
fi

OFRECORD_PATH="train_label"
if [ ! -d "$OFRECORD_PATH" ]; then
    wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/Dataset/Seg/ISBI_Challenge_cell_segmentation/train_label.tar.gz
    tar xvf train_label.tar.gz
fi

CHECKPOINT_PATH="checkpoints"
if [ ! -d "$CHECKPOINT_PATH" ]; then
    mkdir $CHECKPOINT_PATH
fi

DATA_PATH="train_image"
EPOCHS=40
BATCH_SIZE=1


python3 TrainUnetDataSet.py \
    --data_path $DATA_PATH \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \