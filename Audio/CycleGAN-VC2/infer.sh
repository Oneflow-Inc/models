set -aux


PRETRAIN_MODELS='pretrain_models'
DATA_PATH='sample/'

if [ ! -d "$PRETRAIN_MODELS" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/CycleGan-VC2/pretrain_models.zip
    unzip pretrain_models.zip
    rm -fr pretrain_models.zip
fi

if [ ! -d "$DATA_PATH" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/CycleGan-VC2/sample.zip
    unzip sample.zip
    rm -fr sample.zip
fi

echo "Starting infering ..."

python3 infer.py
