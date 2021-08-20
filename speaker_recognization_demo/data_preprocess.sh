set -aux

DATA_DIR="data"
DATA_PREPROCESSED_DIR="data_preprocessed"

if [ ! -d "$DATA_DIR" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/speaker_recognizatian_demo/data.zip
    unzip data.zip
fi

if [ ! -d "$DATA_PREPROCESSED_DIR" ]; then
    mkdir $DATA_PREPROCESSED_DIR
    cd $DATA_PREPROCESSED_DIR
    touch label_dict.json
    mkdir train test && cd ../
fi

python data_preprocess.py
