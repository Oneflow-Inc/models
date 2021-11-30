set -aux

OFRECORD_PATH="predict_image"
if [ ! -d "$OFRECORD_PATH" ]; then
    wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/Dataset/Seg/ISBI_Challenge_cell_segmentation/predict_image.tar.gz
    tar xvf predict_image.tar.gz
fi

python3 predict_unet_test.py