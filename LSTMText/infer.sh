set -aux

SEQUENCE_LEN=128
BATCH_SIZE=1
EMBEDDING_DIM=100
NFC=128
HIDDEN_SIZE=256

LOAD_DIR="pretrain_model.pt"
TEXT="It is awesome! It is nice. The director does a good job!"

python3 infer.py \
    --sequence_length $SEQUENCE_LEN \
    --batch_size $BATCH_SIZE \
    --emb_dim $EMBEDDING_DIM \
    --nfc $NFC \
    --hidden_size $HIDDEN_SIZE \
    --model_load_dir $LOAD_DIR \
    --text "$TEXT"