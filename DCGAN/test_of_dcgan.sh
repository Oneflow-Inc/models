  
set -aux

MODEL_PATH="g_99"
SAVE_PATH="test_images.png"


if [ ! -d "$MODEL_PATH" ]; then
  wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/DCGAN_g.zip
  unzip DCGAN_g.zip 
fi

python3 test_of_dcgan.py --model_path $MODEL_PATH --save_path $SAVE_PATH