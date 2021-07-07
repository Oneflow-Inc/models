set -aux

DATA_PATH="data"
if [ ! -d "$DATA_PATH" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/seq2seq_data.tar.gz
    tar -xzvf seq2seq_data.tar.gz
fi

PRETRAIN_PATH="saving_model_oneflow"
if [ ! -d "$PRETRAIN_PATH" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/nlp/seq2seq_pre_train.tar.gz
    tar -xzvf seq2seq_pre_train.tar.gz
fi

ENCODER_PATH="./saving_model_oneflow/encoder/"
DECODER_PATH="./saving_model_oneflow/decoder/"

DEVICE='cuda'

python3 eval_oneflow.py \
    --encoder_path $ENCODER_PATH \
    --decoder_path $DECODER_PATH \
    --device $DEVICE \