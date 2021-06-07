set -aux

PRETRAIN_MODEL_PATH="mobilenetv2_imagenet_pretrain_model"
IMAGE_PATH="data/fish.jpg"
# IMAGE_PATH="data/tiger.jpg"
# IMAGE_PATH="data/ILSVRC2012_val_00020287.JPEG"

if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
  echo "TODO"
  # wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/resnet50_imagenet_pretrain_model.tar.gz
  # tar zxf resnet50_imagenet_pretrain_model.tar.gz
fi

python3 infer.py --model_path $PRETRAIN_MODEL_PATH --image_path $IMAGE_PATH
