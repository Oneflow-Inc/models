set -aux

MODEL_PATH="checkpoint/"
Data_PATH="dataset"
Model="Shanghai_BestModelA" #choose  from  Shanghai_BestModelA,Shanghai_BestModelB

if [ ! -d "$MODEL_PATH" ]; then
  mkdir ${MODEL_PATH}
fi

if [ ! -d "${MODEL_PATH}${Model}" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/CSRNet/${Model}.rar
  unrar x ${Model}  ${MODEL_PATH}
fi

if [ ! -d "${Data_PATH}" ]; then

  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/CSRNet/Shanghai_dataset.rar
  unrar x Shanghai_dataset
  mv Shanghai_dataset ${Data_PATH}

fi

python3 infer.py 'checkpoint/Shanghai_BestModelA/shanghaiA_bestmodel' 'dataset/part_A_final/test_data/images/IMG_100.jpg' 'dataset/part_A_final/test_data/ground_truth/IMG_100.h5'
