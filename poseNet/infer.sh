set -aux

PRETRAIN_MODEL_PATH="posenet_pretrain_model"
IMAGE_PATH="data/0-1.jpg"

if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/pose/posenet_pretrain_model.tar.gz
  tar zxf posenet_pretrain_model.tar.gz
fi

python3 infer.py --model_path $PRETRAIN_MODEL_PATH --image_path $IMAGE_PATH
