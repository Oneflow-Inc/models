set -aux

REID_MODEL_PATH="reid_oneflow_model"
DATASET_PATH="./datasets"


if [ ! -d "$REID_MODEL_PATH" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/reid/reid_oneflow_model.zip
  unzip reid_oneflow_model.zip
fi


python3 reid.py --evaluate --load_weights $REID_MODEL_PATH --data_dir $DATASET_PATH
