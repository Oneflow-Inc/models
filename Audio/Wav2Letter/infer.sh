set -aux

PRETRAIN_MODEL_PATH="save_models"

if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
    mkdir $PRETRAIN_MODEL_PATH
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/audio/wav2letter.zip
    unzip -d $PRETRAIN_MODEL_PATH ./wav2letter.zip
fi

python infer.py
