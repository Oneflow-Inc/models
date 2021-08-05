set -aux

MODEL_PATH="pretrain_model.pt"
if [ ! -d "$MODEL_PATH" ]; then
    mkdir $MODEL_PATH
fi

BATCH_SIZE=32
EPOCH=30
LEARNING_RATE=3e-4
SEQUENCE_LEN=128
EMBEDDING_DIM=100
NFC=128
HIDDEN_SIZE=256

SAVE_PER_EPOCHS=5
LOAD_DIR="pretrain_model.pt"
SAVE_DIR="pretrain_model.pt"


python3 train_bilstm.py \
    --batch_size $BATCH_SIZE \
    --n_epochs $EPOCH \
    --lr $LEARNING_RATE \
    --sequence_length $SEQUENCE_LEN \
    --emb_dim $EMBEDDING_DIM \
    --nfc $NFC \
    --hidden_size $HIDDEN_SIZE \
    --model_load_dir $LOAD_DIR \
    --model_save_every_n_epochs $SAVE_PER_EPOCHS \
    --model_save_dir $SAVE_DIR