set -aux

TSN_MODEL_PATH="tsn_model_oneflow"
DATASET_PATH="./data"

if [ ! -d "$DATASET_PATH" ]; then
    mkdir data
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/TSN/kinetics400.zip
    unzip -d ./data kinetics400.zip
fi

if [ ! -d "$TSN_MODEL_PATH" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/action_recognition/tsn_model_oneflow.zip
    unzip tsn_model_oneflow.zip
fi

python3 test_recognizer.py --checkpoint $TSN_MODEL_PATH --data_dir $DATASET_PATH
