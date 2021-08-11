set -aux

PRETRAIN_MODEL_PATH="resnet50_imagenet_pretrain_model"
IMAGE_PATH="data/fish.jpg"
<<<<<<< HEAD
# IMAGE_PATH="data/tiger.jpg"
# IMAGE_PATH="data/ILSVRC2012_val_00020287.JPEG"
=======
>>>>>>> 4ccaac3e4b62b12debe261a9a807df9587cb7b87

if [ ! -d "data" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/data.tar.gz
  tar zxf data.tar.gz
fi

if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/resnet50_imagenet_pretrain_model.tar.gz
  tar zxf resnet50_imagenet_pretrain_model.tar.gz
fi

python3 graph/infer.py --model_path $PRETRAIN_MODEL_PATH --image_path $IMAGE_PATH
