set -aux

DATASET_PATH="./data/facades/"
if [ ! -d "$DATASET_PATH" ]; then
    mkdir -p ./data/
    wget -N http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz -O ./data/facades.tar.gz
    tar -zxvf ./data/facades.tar.gz -C ./data/
    rm ./data/facades.tar.gz
fi

LEARNING_RATE=2e-4
EPOCH=200
TRAIN_BATCH_SIZE=1
TRAIN_PATH="./of_pix2pix"

python3 train.py \
    -lr $LEARNING_RATE \
    -e $EPOCH \
    --batch_size $TRAIN_BATCH_SIZE \
    --path $TRAIN_PATH \