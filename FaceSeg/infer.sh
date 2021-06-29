set -aux

PRETRAIN_MODEL_PATH="linknet_oneflow_model"


python3 infer.py --model_path $PRETRAIN_MODEL_PATH