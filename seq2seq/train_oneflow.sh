set -aux


DATA_PATH="data"
if [ ! -d "$DATA_PATH" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/seq2seq_data.tar.gz
    tar -xzvf seq2seq_data.tar.gz
fi

DEVICE='cuda'


python3 train_oneflow.py \
    --device $DEVICE
