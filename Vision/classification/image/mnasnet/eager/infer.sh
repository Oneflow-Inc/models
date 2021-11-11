export PYTHONPATH=$PWD:$PYTHONPATH 
set -aux

PRETRAIN_MODEL_PATH="./weight/mnasnet0_5"
IMAGE_PATH="data/fish.jpg"

if [ ! -d "data" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/data.tar.gz
  tar zxf data.tar.gz
fi

if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
    mkdir weight
    cd weight
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/MNASNet/mnasnet0_5.zip
    unzip mnasnet0_5.zip
    cd ..
fi

python3 eager/infer.py --model_path $PRETRAIN_MODEL_PATH --image_path $IMAGE_PATH