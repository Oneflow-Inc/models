set -aux

MODEL_PATH="best_model"
if [ ! -d "$MODEL_PATH" ]; then
    mkdir $MODEL_PATH
fi

BATCH_SIZE=32
EPOCH=15
LEARNING_RATE=0.0001

SEQUENCE_LEN=128
VOCAB_SZ=100000
D_MODEL=512
DROPOUT=0.1
NHEAD=8
NUM_LAYERS=4
DIM_FF=1024

IMDB_PATH="../datasets/imdb"
LOAD_DIR="."
SAVE_DIR="best_model"

python3 train_transformer_imdb.py \
    --batch_size $BATCH_SIZE \
    --n_epochs $EPOCH \
    --lr $LEARNING_RATE \
    --sequence_len $SEQUENCE_LEN \
    --vocab_sz $VOCAB_SZ \
    --d_model $D_MODEL \
    --dropout $DROPOUT \
    --n_head $NHEAD \
    --n_encoder_layers $NUM_LAYERS \
    --dim_feedforward $DIM_FF \
    --imdb_path "$IMDB_PATH" \
    --load_dir $LOAD_DIR \
    --save_dir $SAVE_DIR