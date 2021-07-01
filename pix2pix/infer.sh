set -aux
MODEL_PATH='pix2pix_g_200'
IMAGE_PATH='./data/facades/test/1.jpg'

if [ ! -d "$MODEL_PATH" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/gan/pix2pix_g_200.zip
  unzip pix2pix_g_200.zip
  rm pix2pix_g_200.zip
fi

python infer.py \
    --model_path $MODEL_PATH \
    --image_path $IMAGE_PATH \
