set -aux

PRETRAIN_MODEL_PATH="resnet50_imagenet_pretrain_model"
DATASET_PATH="../data"


if [ ! -d "PRETRAIN_MODEL_PATH" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/action_recognition/resnet50_imagenet_pretrain_model.zip
  unzip resnet50_imagenet_pretrain_model.zip
fi

python3 train_recognizer.py --pretrained PRETRAIN_MODEL_PATH --data_dir $DATASET_PATH
