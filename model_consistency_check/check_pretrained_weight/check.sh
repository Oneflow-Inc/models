set -aux

OFRECORD_PATH="/data/imagenet/ofrecord"

WEIGHT_STYLE="pytorch"
LOAD_CHECKPOINT="/data/rentianhe/code/models_003/compare_alexnet/models/model_consistency_check/weight/alexnet-owt-7be5be79.pth"
VAL_BATCH_SIZE=512


python3 check_pretrained_weight/check_from_pretrained_weight.py \
    --load_checkpoint $LOAD_CHECKPOINT \
    --ofrecord_path $OFRECORD_PATH \
    --val_batch_size $VAL_BATCH_SIZE \
    --weight_style $WEIGHT_STYLE



