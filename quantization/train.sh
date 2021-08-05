set -aux

OFRECORD_PATH="ofrecord"
if [ ! -d "$OFRECORD_PATH" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenette_ofrecord.tar.gz
    tar zxf imagenette_ofrecord.tar.gz
fi

CHECKPOINT_PATH="checkpoints"
if [ ! -d "$CHECKPOINT_PATH" ]; then
    mkdir $CHECKPOINT_PATH
fi

LEARNING_RATE=0.001
MOM=0.9
EPOCH=20
TRAIN_BATCH_SIZE=16
VAL_BATCH_SIZE=16
QUANTIZATION_BIT=4
QUANTIZATION_SCHEME="symmetric"
QUANTIZATION_FORMULA="google"
PER_LAYER_QUANTIZATION=No
# LOAD PREVIOUS CHECKPOINT 
# LOAD_CHECKPOINT=$CHECKPOINT_PATH/epoch_2_val_acc_0.111168

python3 quantization_aware_training.py \
    --save_checkpoint_path $CHECKPOINT_PATH \
    --ofrecord_path $OFRECORD_PATH \
    --learning_rate $LEARNING_RATE \
    --mom $MOM \
    --epochs $EPOCH \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --val_batch_size $VAL_BATCH_SIZE \
    --quantization_bit $QUANTIZATION_BIT\
    --quantization_scheme $QUANTIZATION_SCHEME\
    --quantization_formula $QUANTIZATION_FORMULA\
    --per_layer_quantization $PER_LAYER_QUANTIZATION
    # --load_checkpoint $LOAD_CHECKPOINT


