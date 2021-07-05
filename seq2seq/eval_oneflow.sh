set -aux

PRETRAIN_PATH="saving_model_oneflow"
if [ ! -d "$PRETRAIN_PATH" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/nlp/seq2seq_pretrain.tar.gz
    unzip seq2seq_pretrain.tar.gz
fi


python3 eval_oneflow.py