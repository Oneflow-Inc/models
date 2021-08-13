set -aux


PRETRAIN_DIR="./roberta_pretrain_oneflow/roberta-base"
KWARGS_PATH="./roberta_pretrain_oneflow/roberta-base/roberta-base.json"
MODEL_LOAD_DIR="./pretrain_model_SST-2"
TEXT="this is junk food cinema at its greasiest ."  
# YOU CAN CHANGE THE 'TEXT' AS YOU WISH, SUCH AS /
#"mark me down as a non-believer in werewolf films that are not serious and rely on stupidity as a substitute for humor ." 
#"most new movies have a bright sheen ."

if [ ! -d "$MODEL_LOAD_DIR" ]; then
    echo "Directory '$MODEL_LOAD_DIR' doesn't exist."
    echo "Please train the model first or download pretrained model."
    exit 1
fi

python3 infer.py \
    --pretrain_dir $PRETRAIN_DIR\
    --kwargs_path $KWARGS_PATH\
    --model_load_dir $MODEL_LOAD_DIR \
    --text "$TEXT" 