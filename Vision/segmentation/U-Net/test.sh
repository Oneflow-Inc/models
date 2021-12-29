set -aux

OFRECORD_PATH="predict_image"
if [ ! -d "$OFRECORD_PATH" ]; then
    wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/Dataset/Seg/ISBI_Challenge_cell_segmentation/predict_image.tar.gz
    tar xvf predict_image.tar.gz
fi


CHECKPOINT='./checkpoints'
TEST_DATA_PATH='test_image/'
SAVE_RES_PATH="./predict_image/test.png"


python3 predict_unet_test.py \
    --checkpoint $CHECKPOINT \
    --Test_Data_path $TEST_DATA_PATH \
    --save_res_path $SAVE_RES_PATH \
