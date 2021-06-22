GPU=0
DATASET="/dataset/coco/train2014"
STYLE_IMAGE="images/style-images/sketch.jpg"
SAVE_DIR="saved_models/"
EPOCHS=2
CUDA=1
LOG_INTERVAL=500
CHECKPOINTS="checkpoints/"
CHECKPOINT_INTERVAL=2000
LR=0.01
CONTENT_WEIGHT=50000

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
    --content-weight $CONTENT_WEIGHT



