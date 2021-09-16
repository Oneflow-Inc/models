set -aux

SEQUENCE_LEN=128
EMBEDDING_DIM=100
HIDDEN_SIZE=256

LOAD_DIR="pretrain_model"
IMDB_PATH="./imdb"
TEXT="It is awesome! It is nice. The director does a good job!"

if [ ! -d "$LOAD_DIR" ]; then
    echo "Directory '$LOAD_DIR' doesn't exist."
    echo "Please train the model first or download pretrained model."
    exit 1
fi

if [ ! -d "$IMDB_PATH" ]; then
    echo "Directory '$IMDB_PATH' doesn't exist."
    echo "Please set the correct path of imdb dataset."
    exit 1
fi

python3 infer.py \
    --sequence_length $SEQUENCE_LEN \
    --emb_dim $EMBEDDING_DIM \
    --hidden_size $HIDDEN_SIZE \
    --model_load_dir $LOAD_DIR \
    --imdb_path $IMDB_PATH \
    --text "$TEXT"
