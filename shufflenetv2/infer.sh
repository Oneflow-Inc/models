set -aux

PRETRAIN_MODEL_PATH="shufflenet_imagenet_pretrain_model/"
MODEL="shufflenetv2_x1.0" #can be one of shufflenetv2_x0.5, shufflenetv2_x1.0
IMAGE_PATH="data/tiger.jpg"

if [ ! -d "data" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/data.tar.gz
  tar zxf data.tar.gz
fi

if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
  mkdir ${PRETRAIN_MODEL_PATH}
fi

if [ ! -d "${PRETRAIN_MODEL_PATH}oneflow_${MODEL}" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/shufflenetv2/oneflow_${MODEL}.zip
  unzip oneflow_${MODEL}.zip -d ${PRETRAIN_MODEL_PATH}
fi

python3 infer.py --model_path ${PRETRAIN_MODEL_PATH}oneflow_${MODEL} --image_path $IMAGE_PATH --model $MODEL
