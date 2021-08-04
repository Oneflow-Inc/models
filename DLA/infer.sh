set -aux

PRETRAIN_MODEL_PATH="pretrain_model/epoch_1396_val_acc_0.921875"
IMAGE_PATH="data/bus.jpg"
# IMAGE_PATH="data/tiger.jpg"
# IMAGE_PATH="data/ILSVRC2012_val_00020287.JPEG"

if [ ! -d "data" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/DLA/data.zip
  unzip data.zip
fi

if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/DLA/pretrain_model.zip
  unzip pretrain_model.zip
fi

python3 infer.py --model_path $PRETRAIN_MODEL_PATH --image_path $IMAGE_PATH









