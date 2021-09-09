set -aux


OPTIONS_PATH='cfg/AM_MobileNet1D_TIMIT.cfg'
PRETRAIN_MODELS='pretrain_models'

if [ ! -d "$PRETRAIN_MODELS" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/audio/AM_MobileNet1D.zip
    unzip AM_MobileNet1D.zip
    rm -fr AM_MobileNet1D.zip
fi

python3 infer.py --cfg $OPTIONS_PATH
