set -aux

PRETRAIN_MODEL_PATH="resnet50_pretrained_model"
DATASET_PATH="./"

if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/reid/resnet50_pretrained_model.zip
  unzip resnet50_pretrained_model.zip
fi

if [ ! -d "$DATASET_PATH" ]; then
  wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/Dataset/market1501.zip
  unzip market1501.zip
fi

python3  reid.py --load_weights $PRETRAIN_MODEL_PATH --data_dir $DATASET_PATH
