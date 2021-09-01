set -aux

LEARNING_RATE=0.001
MOM=0.9
EPOCH=10
MODEL_PATH="./pretrained_resnet_oneflow_model"
DATASET_PATH="./faceseg_data/"
SAVE_MODEL_NAME="linknet_oneflow_model"
# LOAD PREVIOUS CHECKPOINT
# LOAD_CHECKPOINT="/PATH/TO/CHECKPOINT"

if [ ! -d "$DATASET_PATH" ]; then
    wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/faceseg_data.zip
    mkdir faceseg_data
    unzip faceseg_data -d faceseg_data
fi

python3 train.py \
    --learning_rate $LEARNING_RATE \
    --mom $MOM \
    --epochs $EPOCH \
    --model_path $MODEL_PATH \
    --dataset_path $DATASET_PATH \
    --save_model_name $SAVE_MODEL_NAME \
    # --load_checkpoint $LOAD_CHECKPOINT
