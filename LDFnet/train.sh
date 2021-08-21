set -aux

cd ./model

PRETRAIN_MODEL_PATH="./pretrained/resnet50-19c8e357"
DATASET_PATH="./data/DUTS-TR"

if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/seg/LDFnet/resnet50-19c8e357.zip
  unzip resnet50-19c8e357.zip
fi

python3  train.py 
