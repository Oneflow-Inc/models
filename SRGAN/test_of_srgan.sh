set -aux

DATA_PATH='data/'
IMAGE_NAME='monarch'
MODEL_PATH="SRGAN_netG_epoch_4_99"

if [ ! -d $DATA_PATH ]; then
  mkdir ${DATA_PATH}
fi
if [ ! -f $DATA_PATH$IMAGE_NAME'.png' ]; then
  wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/monarch.png
  mv $IMAGE_NAME'.png' $DATA_PATH
fi
if [ ! -f $DATA_PATH$IMAGE_NAME'x4.png' ]; then
  wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/monarchx4.png
  mv $IMAGE_NAME'x4.png' $DATA_PATH
fi


if [ ! -d "$MODEL_PATH" ]; then
  wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/SRGAN_netG_epoch_4_99.zip
  unzip SRGAN_netG_epoch_4_99.zip
fi

python3 test_of_srgan.py --image_path $DATA_PATH$IMAGE_NAME'x4.png' --hr_path  $DATA_PATH$IMAGE_NAME'.png' --model_path $MODEL_PATH --save_image  $DATA_PATH$IMAGE_NAME'-oneflow.png'
