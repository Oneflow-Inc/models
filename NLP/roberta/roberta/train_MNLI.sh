set -aux

BATCH_SIZE=32
EPOCH=20
LEARNING_RATE=1e-5
PRETRAIN_DIR="./roberta-base-oneflow/weights"
KWARGS_PATH="./roberta-base-oneflow/parameters.json"
SAVE_DIR="./pretrain_model_MNLI"
TASK="MNLI"

if [ ! -d $SAVE_DIR ]; then
    mkdir $SAVE_DIR
fi

python3 train_MNLI.py \
    --batch_size $BATCH_SIZE \
    --n_epochs $EPOCH \
    --lr $LEARNING_RATE \
    --pretrain_dir $PRETRAIN_DIR\
    --kwargs_path $KWARGS_PATH\
    --model_save_dir $SAVE_DIR\
    --task $TASK 