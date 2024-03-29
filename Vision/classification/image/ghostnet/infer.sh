set -aux

PRETRAIN_MODEL_PATH="ghostnet_imagenet_pretrain_model/"
MODEL="ghostnet" #choose from ghostnet

# IMAGE_PATH="data/fish.jpg"
IMAGE_PATH="data/tiger.jpg"
# IMAGE_PATH="data/ILSVRC2012_val_00020287.JPEG"

if [ ! -d "data" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/data.tar.gz
  tar zxf data.tar.gz
fi

if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
  mkdir ${PRETRAIN_MODEL_PATH}
fi

if [ ! -d "${PRETRAIN_MODEL_PATH}/checkpoints/epoch_484_val_acc_0.908418" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/ghostnet/ghostnet_oneflow_model.tar.gz
  tar zxf ghostnet_oneflow_model.tar.gz --directory ${PRETRAIN_MODEL_PATH}
fi

python3 infer.py --model_path ${PRETRAIN_MODEL_PATH}/checkpoints/epoch_484_val_acc_0.908418 --image_path $IMAGE_PATH --model $MODEL
