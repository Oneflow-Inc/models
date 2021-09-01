set -aux

PRETRAIN_MODEL_PATH="./scnet_oneflow_model"
IMAGE_PATH="data/img_red.png"

if [ ! -f "$IMAGE_PATH" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/scnet/data.zip
  unzip data.zip
fi

if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/scnet/scnet_oneflow_model.zip
  unzip scnet_oneflow_model.zip
fi

python3 infer.py --model_path $PRETRAIN_MODEL_PATH --image_path $IMAGE_PATH
