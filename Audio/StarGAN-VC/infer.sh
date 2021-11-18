set -aux


PRETRAIN_MODELS='pretrain_models'

if [ ! -d "$PRETRAIN_MODELS" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/StarGAN-VC/pretrain_models.zip
    unzip pretrain_models.zip
    rm -fr pretrain_models.zip
fi

echo "Starting infering ..."

python3 main.py \
    --mode test \
    --src_speaker TM1 --trg_speaker "['TM1','SF1']"
