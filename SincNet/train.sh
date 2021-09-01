set -aux


DATA_PATH='data/'
DATA_LISTS='data_lists'
TIMIT_FOLDER='data/lisa/data/timit/raw/TIMIT'
DATA_PREPARED='data_preprocessed'
LABEL='data_lists/TIMIT_all_upper.scp'
OPTIONS_PATH='cfg/SincNet_TIMIT.cfg'

if [ ! -d "$DATA_PATH" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/SincNet/TIMIT.zip
    unzip TIMIT.zip
    rm -fr TIMIT.zip
fi

if [ ! -d "$DATA_LISTS" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/SincNet/data_lists.zip
    unzip data_lists.zip
    rm -fr data_lists.zip
fi

if [ ! -d "$DATA_PREPARED" ]; then
    python utils/TIMIT_preparation.py $TIMIT_FOLDER $DATA_PREPARED $LABEL
fi

echo "Data prepared !"
echo "Starting training ..."

python train.py --cfg $OPTIONS_PATH
