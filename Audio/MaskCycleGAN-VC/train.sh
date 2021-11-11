set -aux


DATA_PATH='vcc2018/'
DATA_PREPARED='vcc2018_preprocessed/'

if [ ! -d "$DATA_PATH" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/MaskCycleGAN-VC/vcc2018.zip
    unzip vcc2018.zip
    rm -fr vcc2018.zip
fi

if [ ! -d "$DATA_PREPARED" ]; then
    python3 utils/data_preprocess.py \
        --data_directory vcc2018/vcc2018_training \
        --preprocessed_data_directory vcc2018_preprocessed/vcc2018_training \
        --speaker_ids VCC2SF1 VCC2SF2 VCC2SF3 VCC2SF4 VCC2SM1 VCC2SM2 VCC2SM3 VCC2SM4 VCC2TF1 VCC2TF2 VCC2TM1 VCC2TM2

    python3 utils/data_preprocess.py \
        --data_directory vcc2018/vcc2018_evaluation \
        --preprocessed_data_directory vcc2018_preprocessed/vcc2018_evaluation \
        --speaker_ids VCC2SF1 VCC2SF2 VCC2SF3 VCC2SF4 VCC2SM1 VCC2SM2 VCC2SM3 VCC2SM4 VCC2TF1 VCC2TF2 VCC2TM1 VCC2TM2
fi

echo "Data prepared !"
echo "Starting training ..."

python3 train.py \
    --seed 0 \
    --preprocessed_data_dir vcc2018_preprocessed/vcc2018_training \
    --speaker_A_id VCC2SF3 \
    --speaker_B_id VCC2TF1 \
    --epochs_per_save 1000 \
    --num_epochs 6172 \
    --decay_after 2e5 \
    --batch_size 9 \
    --sample_rate 22050 \
    --num_frames 64 \
    --max_mask_len 25 \
