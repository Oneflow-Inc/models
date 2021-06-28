set -aux

PRETRAIN_MODEL_PATH="resnet50_pretrained_model"
DATASET_PATH="./dataset"

if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/reid/resnet50_pretrained_model.zip
  unzip resnet50_pretrained_model.zip
fi

if [ ! -d "$DATASET_PATH" ]; then
  mkdir $DATASET_PATH
  cd $DATASET_PATH
  wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/Dataset/market1501.zip
  unzip market1501.zip
  cd ..
fi

python3  reid.py --load_weights $PRETRAIN_MODEL_PATH --data_dir $DATASET_PATH
