set -aux

SEQUENCE_LEN=128
VOCAB_SZ=100000
D_MODEL=512
DROPOUT=0.1
NHEAD=8
NUM_LAYERS=4
DIM_FF=1024
LOAD_DIR="best_model"
IMDB_PATH="../datasets/imdb"
TEXT="It is awesome! It is nice. The director does a good job!"

if [ ! -d "$LOAD_DIR" ]; then
    echo "Directory '$LOAD_DIR' doesn't exist."
    echo "Please train the model first or download pretrained model."
    exit 1
fi

python3 infer_transformer_imdb.py \
    --sequence_len $SEQUENCE_LEN \
    --vocab_sz $VOCAB_SZ \
    --d_model $D_MODEL \
    --dropout $DROPOUT \
    --n_head $NHEAD \
    --n_encoder_layers $NUM_LAYERS \
    --dim_feedforward $DIM_FF \
    --load_dir $LOAD_DIR \
    --imdb_path "$IMDB_PATH" \
    --text "$TEXT"