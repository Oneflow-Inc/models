set -aux

IMAGE_PATH="data/fish.jpg"
# Note that We only provide pretrained model weight under image_size=384
# Model Arch: ['vit_b_16_384', 'vit_b_32_384', 'vit_l_16_384', 'vit_l_32_384']
PRETRAIN_MODEL_PATH="./vit_b_16_384"
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
