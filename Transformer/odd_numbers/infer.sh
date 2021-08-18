set -aux

VOCAB_SZ=10000
D_MODEL=512
DROPOUT=0.0
NHEAD=2
NUM_ENCODER_LAYERS=1
NUM_DECODER_LAYERS=1
DIM_FF=128

LOAD_DIR="best_model"
INPUT_START=4386

if [ ! -d "$LOAD_DIR" ]; then
    echo "Directory '$LOAD_DIR' doesn't exist."
    echo "Please train the model first or download pretrained model."
    exit 1
fi

python3 infer_transformer_odd_numbers.py \
    --vocab_sz $VOCAB_SZ \
    --d_model $D_MODEL \
    --dropout $DROPOUT \
    --n_head $NHEAD \
    --n_encoder_layers $NUM_ENCODER_LAYERS \
    --n_decoder_layers $NUM_DECODER_LAYERS \
    --dim_feedforward $DIM_FF \
    --load_dir $LOAD_DIR \
    --input_start $INPUT_START
