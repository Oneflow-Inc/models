set -aux

PRETRAIN_MODEL_PATH="./mobilenetv3_oneflow_model"
IMAGE_PATH="data/fish.jpg"
# IMAGE_PATH="data/tiger.jpg"
# IMAGE_PATH="data/ILSVRC2012_val_00020287.JPEG"

if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/mobilenetv3/mobilenetv3_oneflow_model.tar.gz
  tar -xzvf mobilenetv3_oneflow_model.tar.gz
fi

python3 infer.py --model_path $PRETRAIN_MODEL_PATH --image_path $IMAGE_PATH
