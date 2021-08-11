LEARNING_RATE=3e-2
WEIGHT_DECAY=0.0
DATA_PATH='/data/imagenet/ofrecord/'
PRETRAINED_DIR="/data/rentianhe/weight/vit_oneflow/ViT-B_16_oneflow"
TRAIN_BATCH_SIZE=512
ACCU_STEPS=32
EVAL_BATCH_SIZE=32
NUM_STEPS=20000
WARMUP_STEPS=500
NAME='test'


CUDA_VISIBLE_DEVICES=1 python3 train.py \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --data_path $DATA_PATH \
    --pretrained_dir $PRETRAINED_DIR \
    --gradient_accumulation_steps $ACCU_STEPS \
    --num_steps $NUM_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --name $NAME \

