set -aux


OPTIONS_PATH='cfg/SincNet_TIMIT.cfg'
PRETRAIN_MODELS='pretrain_models'

if [ ! -d "$PRETRAIN_MODELS" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/audio/SincNet.zip
    unzip SincNet.zip
    rm -fr SincNet.zip
fi

python infer.py --cfg $OPTIONS_PATH
