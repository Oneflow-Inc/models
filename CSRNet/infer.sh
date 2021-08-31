set -aux

PRETRAIN_MODEL_PATH="vgg_imagenet_pretrain_model"
MODEL_PATH="checkpoint/"
Data_PATH="dataset"
Model="Shanghai_BestModelA" # choose from {'Shanghai_BestModelA', 'Shanghai_BestModelB'}


if [ ! -d "$MODEL_PATH" ]; then
  mkdir ${MODEL_PATH}
fi
if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
  mkdir ${PRETRAIN_MODEL_PATH}
  cd ${PRETRAIN_MODEL_PATH}
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/vgg_models/vgg16_oneflow_model.tar.gz
  tar -zxvf vgg16_oneflow_model.tar.gz
  cd ..
fi

if [ ! -d "${MODEL_PATH}${Model}" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/CSRNet/${Model}.zip
  unzip ${Model}.zip 
  mv ${Model} ${MODEL_PATH}
fi

if [ ! -d "${Data_PATH}" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/CSRNet/Shanghai_dataset.zip
  unzip Shanghai_dataset.zip
  mv Shanghai_dataset ${Data_PATH}
fi

python3 infer.py 'checkpoint/Shanghai_BestModelA/shanghaiA_bestmodel' 'dataset/part_A_final/test_data/images/IMG_100.jpg' 'dataset/part_A_final/test_data/ground_truth/IMG_100.h5'
