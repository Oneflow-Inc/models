set -aux


OPTIONS_PATH='configs/lstm.yaml'

CUDA_VISIBLE_DEVICES=2 python predict.py --config $OPTIONS_PATH
