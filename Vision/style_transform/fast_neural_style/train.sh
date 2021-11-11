PRETRAIN_MODEL_PATH="vgg_imagenet_pretrain_model/"
MODEL="vgg16" # choose from vgg16 and vgg19

if [ ! -d "images" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/fast_neural_style/images.tar.gz
  tar zxf images.tar.gz
fi

if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
  mkdir ${PRETRAIN_MODEL_PATH}
fi

if [ ! -d "${PRETRAIN_MODEL_PATH}${MODEL}_oneflow_model" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/vgg_models/${MODEL}_oneflow_model.tar.gz
  tar zxf ${MODEL}_oneflow_model.tar.gz --directory ${PRETRAIN_MODEL_PATH}
fi

GPU=0 # specify GPU
DATASET="/dataset/coco/train2014/" # dataset directory
STYLE_IMAGE="images/style-images/sketch.jpeg" # style image for training
SAVE_DIR="saved_models/" # directory to save the final model
EPOCHS=2 # number of epochs
CUDA=1 # use CUDA or not
LOG_INTERVAL=100 # log interval
CHECKPOINT_DIR="checkpoints/" # checkpoints directory
# CHECKPOINT="checkpoints/SKETCHCROP-3_CW_30000_lr_0.001ckpt_sketch_epoch0_4000" # uncomment this line if you want to load from a checkpoint
CHECKPOINT_INTERVAL=200 # checkpoint interval
LR=0.001 # learning rate
CONTENT_WEIGHT=30000 # tune this
STYLE_DIR="style_log/"

if [ ! -d "$SAVE_DIR" ]; then
  mkdir ${SAVE_DIR}
fi

if [ ! -d "$CHECKPOINT_DIR" ]; then
  mkdir ${CHECKPOINT_DIR}
fi

if [ ! -d "$STYLE_DIR" ]; then
  mkdir ${STYLE_DIR}
fi

echo "training with learning rate $LR and content weight $CONTENT_WEIGHT in $MODEL"
# Delete CUDA_VISIBLE_DEVICES=${GPU} if you are using cpu
CUDA_VISIBLE_DEVICES=${GPU} python3 neural_style/neural_style.py train \
    --dataset ${DATASET} \
    --style-image $STYLE_IMAGE \
    --save-model-dir $SAVE_DIR \
    --epochs $EPOCHS \
    --cuda $CUDA \
    --log-interval $LOG_INTERVAL\
    --checkpoint-model-dir $CHECKPOINT_DIR\
    --checkpoint-interval $CHECKPOINT_INTERVAL\
    --lr $LR \
    --content-weight $CONTENT_WEIGHT \
    --vgg $MODEL \
    --style-log-dir $STYLE_DIR \
    # --load-checkpoint-dir $CHECKPOINT 
    # Uncomment the line above if you are fine-tuning or resuming training



