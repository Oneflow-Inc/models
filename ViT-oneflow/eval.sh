set -aux

DEFAULT_WEIGHT_PATH="weight/ViT-B_16_oneflow"
if [ ! -d "$DEFAULT_WEIGHT_PATH" ]; then
    mkdir weight
    cd weight
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/ViT-OneFlow/ViT-B_16_oneflow.zip
    unzip ViT-B_16_oneflow.zip
    cd ..
fi


CHECKPOINT_PATH="./weight/ViT-B_16_oneflow"
BATCH_SIZE=32
IMAGE_SIZE=384
NUM_WORKERS=8
DATA_DIR="/data/imagenet/extract"


python3 eval.py \
    --checkpoint-path $CHECKPOINT_PATH \
    --image-size $IMAGE_SIZE \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --data-dir $DATA_DIR \



