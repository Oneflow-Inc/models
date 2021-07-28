set -aux

PRETRAIN_MODEL_PATH="scnet_acc_0.947254"
IMAGE_PATH="data/img_red.png"

if [ ! -d "$IMAGE_PATH" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/scnet/data.zip
  unzip data.zip
fi

python3 infer.py --model_path $PRETRAIN_MODEL_PATH --image_path $IMAGE_PATH
