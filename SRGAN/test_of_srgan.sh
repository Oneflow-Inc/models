  
set -aux

IMAGE_NAME='data/Set14/LR_bicubic/X4/monarchx4.png'
HR_NAME='data/Set14/HR/monarch.png'
SAVE_IMAGE='data/Set14/SR/monarchx4-oneflow.png'
MODEL_NAME="netG_epoch_4_99"


if [ ! -d "$MODEL_NAME" ]; then
  wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/SRGAN_netG_epoch_4_99.zip
  unzip SRGAN_netG_epoch_4_99.zip
fi

python3 test_of_srgan.py --image_name $IMAGE_NAME --hr_name $HR_NAME --model_name $MODEL_NAME --save_image $SAVE_IMAGE
