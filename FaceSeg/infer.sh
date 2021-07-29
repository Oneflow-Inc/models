set -aux

PRETRAIN_MODEL_PATH="linknet_oneflow_model"

if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/seg/linknet_oneflow_model.zip
  unzip linknet_oneflow_model.zip
  rm linknet_oneflow_model.zip
fi

SOURCE="./source1.avi"
if [ ! -f "$SOURCE" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/seg/source1.avi
fi

python3 infer.py --model_path $PRETRAIN_MODEL_PATH