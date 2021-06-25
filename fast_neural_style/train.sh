PRETRAIN_MODEL_PATH="vgg_imagenet_pretrain_model/"
MODEL="vgg16" # choose from vgg16 and vgg19

if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
  mkdir ${PRETRAIN_MODEL_PATH}
fi

if [ ! -d "${PRETRAIN_MODEL_PATH}${MODEL}_oneflow_model" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/vgg_models/${MODEL}_oneflow_model.tar.gz
  tar zxf ${MODEL}_oneflow_model.tar.gz --directory ${PRETRAIN_MODEL_PATH}
fi

GPU=3
DATASET="/dataset/coco2014/train2014/"
STYLE_IMAGE="images/style-images/sketch-bottle.jpeg"
SAVE_DIR="saved_models/"
EPOCHS=10
CUDA=1
LOG_INTERVAL=100
CHECKPOINTS="checkpoints/"
CHECKPOINT_INTERVAL=500
LR=0.001
CONTENT_WEIGHT=20000

echo "training with learning rate $LR and content weight $CONTENT_WEIGHT in $MODEL"

CUDA_VISIBLE_DEVICES=${GPU} python3 neural_style/neural_style.py train \
    --dataset ${DATASET} \
    --style-image $STYLE_IMAGE \
    --save-model-dir $SAVE_DIR \
    --epochs $EPOCHS \
    --cuda $CUDA \
    --log-interval $LOG_INTERVAL\
    --checkpoint-model-dir $CHECKPOINTS\
    --checkpoint-interval $CHECKPOINT_INTERVAL\
    --lr $LR \
    --content-weight $CONTENT_WEIGHT \
    --vgg $MODEL



