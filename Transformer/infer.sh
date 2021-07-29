set -aux

SEQUENCE_LEN=128
VOCAB_SZ=100000
D_MODEL=512
DROPOUT=0.1
NHEAD=8
NUM_LAYERS=6
DIM_FF=1024
LOAD_DIR="best_model"
TEXT="This film is too bad."

python3 infer_transformer.py \
    --sequence_len $SEQUENCE_LEN \
    --vocab_sz $VOCAB_SZ \
    --d_model $D_MODEL \
    --dropout $DROPOUT \
    --n_head $NHEAD \
    --n_encoder_layers $NUM_LAYERS \
    --dim_feedforward $DIM_FF \
    --load_dir $LOAD_DIR \
    --text "$TEXT"