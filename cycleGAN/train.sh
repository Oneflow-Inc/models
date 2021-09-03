set -aux

DATASET_DIR="./datasets/"

if [ ! -d "$DATASET_DIR" ]; then
    mkdir $DATASET_DIR
fi

CHECKPOINT_SAVE_DIR="./checkpoints/"

if [ ! -d "$CHECKPOINT_SAVE_DIR" ]; then
    mkdir $CHECKPOINT_SAVE_DIR
fi

TRAIN_DATASET="summer2winter_yosemite"
#choose between apple2orange, horse2zebra, summer2winter_yosemite. Note that you need to download the corresponding dataset first.

TRAIN_DATASET_A="./datasets/${TRAIN_DATASET}/trainA/"
TRAIN_DATASET_B="./datasets/${TRAIN_DATASET}/trainB/"

RESIZE_AND_CROP=True
CROP_SIZE=256
LOAD_SIZE=286

BETA1=0.5
BETA2=0.9

SAVE_TMP_IMAGE_PATH="./training_cyclegan_${TRAIN_DATASET}_${BETA1}_${BETA2}/"

if [ ! -d "$SAVE_TMP_IMAGE_PATH" ]; then
    mkdir $SAVE_TMP_IMAGE_PATH
fi

EPOCH=300
LR=0.0001

LAMBDA_A=1
LAMBDA_B=1
LAMBDA_IDENTITY=0.5
 
LOAD_EPOCH=10

echo "Train with Adam beta1 ${BETA1} and beta2 ${BETA2}"

CUDA_VISIBLE_DEVICES=1 python3 train.py \
    --dataset_name $TRAIN_DATASET \
    --datasetA_path $TRAIN_DATASET_A \
    --datasetB_path $TRAIN_DATASET_B \
    --resize_and_crop $RESIZE_AND_CROP \
    --crop_size $CROP_SIZE \
    --load_size $LOAD_SIZE \
    --save_tmp_image_path $SAVE_TMP_IMAGE_PATH \
    --train_epoch $EPOCH \
    --learning_rate $LR \
    --lambda_A $LAMBDA_A \
    --lambda_B $LAMBDA_B \
    --lambda_identity $LAMBDA_IDENTITY \
    --checkpoint_save_dir $CHECKPOINT_SAVE_DIR \
    --checkpoint_load_epoch $LOAD_EPOCH \
    --beta1 $BETA1 \
    --beta2 $BETA2 \
    # --checkpoint_load_dir $CHECKPOINT_LOAD_DIR

