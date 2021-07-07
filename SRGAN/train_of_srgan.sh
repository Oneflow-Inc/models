set -aux

DATA_PATH='data/'
mkdir -p ${DATA_PATH}
wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/VOC2012.zip
unzip VOC2012.zip -d $DATA_PATH

PRETRAIN_MODEL_PATH="vgg_imagenet_pretrain_model/"
MODEL="vgg16" #choose from vgg16, vgg16_bn, vgg19, vgg19_bn
if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
  mkdir ${PRETRAIN_MODEL_PATH}
fi
if [ ! -d "${PRETRAIN_MODEL_PATH}${MODEL}_oneflow_model" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/vgg_models/${MODEL}_oneflow_model.tar.gz
  tar zxf ${MODEL}_oneflow_model.tar.gz --directory ${PRETRAIN_MODEL_PATH}
fi

LEARNING_RATE=0.0002
WEIGHT_DECAY_A=0.5
WEIGHT_DECAY_B=0.999
EPOCH=100
BATCH_SIZE=64
HR_SIZE=88


python3 train_of_srgan.py \
    --lr $LEARNING_RATE \
    --b1 $WEIGHT_DECAY_A \
    --b2 $WEIGHT_DECAY_B \
    --num_epochs $EPOCH \
    --batch_size $BATCH_SIZE \
    --data_dir $DATA_PATH'VOC' \
    --vgg_path ${PRETRAIN_MODEL_PATH}${MODEL}_oneflow_model \
    --hr_size $HR_SIZE 