set -aux

PRETRAIN_MODEL_PATH="./vit_b_16_384"
IMAGE_PATH="data/fish.jpg"
MODEL_ARCH="vit_b_16_384"
IMAGE_SIZE=384


if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/ViT-OneFlow/vit_b_16_384.zip
  unzip vit_b_16_384.zip
fi

if [ ! -d "data" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/data.tar.gz
  tar zxf data.tar.gz
fi

python3 infer.py --model_path $PRETRAIN_MODEL_PATH --image_path $IMAGE_PATH --model_arch $MODEL_ARCH --image_size $IMAGE_SIZE
