set -aux

MODEL_PATH="best_model"
if [ ! -d "$MODEL_PATH" ]; then
    mkdir $MODEL_PATH
fi

BATCH_SIZE=128
EPOCH=30
LEARNING_RATE=0.0001

VOCAB_SZ=10000
D_MODEL=512
DROPOUT=0.0
NHEAD=2
NUM_ENCODER_LAYERS=1
NUM_DECODER_LAYERS=1
DIM_FF=128

LOAD_DIR="."
SAVE_DIR="best_model"

python3 train_transformer_odd_numbers.py \
    --batch_size $BATCH_SIZE \
    --n_epochs $EPOCH \
    --lr $LEARNING_RATE \
    --vocab_sz $VOCAB_SZ \
    --d_model $D_MODEL \
    --dropout $DROPOUT \
    --n_head $NHEAD \
    --n_encoder_layers $NUM_ENCODER_LAYERS \
    --n_decoder_layers $NUM_DECODER_LAYERS \
    --dim_feedforward $DIM_FF \
    --load_dir $LOAD_DIR \
    --save_dir $SAVE_DIR
