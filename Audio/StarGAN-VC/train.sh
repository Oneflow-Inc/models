set -aux


TRAIN_DATA='vcc2016_training.zip'
TEST_DATA='evaluation_all.zip'

if [ ! -f "$TRAIN_DATA" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/StarGAN-VC/vcc2016_training.zip
fi

if [ ! -f "$TEST_DATA" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/StarGAN-VC/evaluation_all.zip
fi

python3 utils/preprocess.py

echo "Data prepared !"
echo "Starting training ..."

python3 main.py
