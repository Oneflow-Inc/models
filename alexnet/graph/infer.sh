set -aux

PRETRAIN_MODEL_PATH="./alexnet_oneflow_model"
IMAGE_PATH="data/fish.jpg"

if [ ! -d "data" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/data.tar.gz
  tar zxf data.tar.gz
fi

if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/alexnet/alexnet_oneflow_model.tar.gz
  tar -xzvf alexnet_oneflow_model.tar.gz
fi

python3 graph/infer.py --model_path $PRETRAIN_MODEL_PATH --image_path $IMAGE_PATH
