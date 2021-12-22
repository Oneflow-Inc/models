set -aux


PRETRAIN_DIR="./flow_roberta-base/weights"
KWARGS_PATH="./flow_roberta-base/parameters.json"
MODEL_LOAD_DIR="./pretrain_model_MNLI"
TEXT="The new rights are nice enough. Everyone really likes the newest benefits."  
TASK="MNLI"
# YOU CAN CHANGE THE 'TEXT' AS YOU WISH

if [ ! -d "$MODEL_LOAD_DIR" ]; then
    echo "Directory '$MODEL_LOAD_DIR' doesn't exist."
    echo "Please train the model first or download pretrained model."
    exit 1
fi

python3 infer_MNLI.py \
    --pretrain_dir $PRETRAIN_DIR\
    --kwargs_path $KWARGS_PATH\
    --model_load_dir $MODEL_LOAD_DIR \
    --text "$TEXT" \
    --task $TASK