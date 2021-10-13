set -aux


DATA_PATH='data/'
DATA_PREPARED='cache/'

if [ ! -d "$DATA_PATH" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/CycleGan-VC2/data.zip
    unzip data.zip
    rm -fr data.zip
fi

if [ ! -d "$DATA_PREPARED" ]; then
    python3 utils/data_preprocess.py
fi

echo "Data prepared !"
echo "Starting training ..."

python3 train.py
