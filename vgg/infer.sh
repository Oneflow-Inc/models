set -aux

PRETRAIN_MODEL_PATH="vgg_imagenet_pretrain_model/"
IMAGE_PATH="data/fish.jpg"
MODEL="vgg16" #choose from vgg16, vgg16_bn, vgg19, vgg19_bn
# IMAGE_PATH="data/tiger.jpg"
# IMAGE_PATH="data/ILSVRC2012_val_00020287.JPEG"

if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
  mkdir ${PRETRAIN_MODEL_PATH}
fi

if [ ! -d "${PRETRAIN_MODEL_PATH}${MODEL}_oneflow_model" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/vgg_models/${MODEL}_oneflow_model.tar.gz
  tar zxf ${MODEL}_oneflow_model.tar.gz --directory ${PRETRAIN_MODEL_PATH}
fi

python3 infer.py --model_path $PRETRAIN_MODEL_PATH --image_path $IMAGE_PATH --model $MODEL
