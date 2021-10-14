set -aux


PRETRAIN_MODELS='pretrain_models'
DATA_PATH='sample/'

if [ ! -d "$PRETRAIN_MODELS" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/MaskCycleGAN-VC/pretrain_models.zip
    unzip pretrain_models.zip
    rm -fr pretrain_models.zip
fi

if [ ! -d "$DATA_PATH" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/MaskCycleGAN-VC/sample.zip
    unzip sample.zip
    rm -fr sample.zip
fi

echo "Starting infering ..."

python3 infer.py \
    --pretrain_models pretrain_models \
    --infer_data_dir sample \
    --decay_after 2e5 \
    --batch_size 9 \
    --sample_rate 22050 \
    --num_frames 64 \
    --max_mask_len 25
