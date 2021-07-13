set -aux

IMAGE_PATH='data/monarchx4.png'
HR_PATH='data/monarch.png'
SAVE_IMAGE_PATH='data/monarchx4-oneflow.png'
MODEL_PATH="SRGAN_netG_epoch_4_99"

DATA_PATH='data/'
mkdir -p ${DATA_PATH}
wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/monarch.png
wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/monarchx4.png
mv "monarch.png" $DATA_PATH
mv "monarchx4.png" $DATA_PATH

if [ ! -d "$MODEL_PATH" ]; then
  wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/SRGAN_netG_epoch_4_99.zip
  unzip SRGAN_netG_epoch_4_99.zip
fi

python3 test_of_srgan.py --image_path $IMAGE_PATH --hr_path $HR_PATH --model_path $MODEL_PATH --save_image $SAVE_IMAGE_PATH
