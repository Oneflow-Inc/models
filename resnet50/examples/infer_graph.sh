# set -aux

PRETRAIN_MODEL_PATH="resnet50_imagenet_pretrain_model"
IMAGE_PATH="data/fish.jpg"

if [ ! -d "data" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/data.tar.gz
  tar zxf data.tar.gz
fi

if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/resnet50_imagenet_pretrain_model.tar.gz
  tar zxf resnet50_imagenet_pretrain_model.tar.gz
fi

# SRC_DIR=/path/to/models/resnet50
SRC_DIR=$(realpath $(dirname $0)/..)

python3 $SRC_DIR/infer.py \
    --model $PRETRAIN_MODEL_PATH \
    --image $IMAGE_PATH \
    --graph \
