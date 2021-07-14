set -aux

TSN_MODEL_PATH="tsn_oneflow_model"
DATASET_PATH="../data"

if [ ! -d "TSN_MODEL_PATH" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/action_recognition/tsn_model_oneflow.zip
  unzip tsn_oneflow_model.zip
fi

python3 test_recognizer.py --checkpoint TSN_MODEL_PATH --data_dir $DATASET_PATH
