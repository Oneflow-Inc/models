set -aux


OPTIONS_PATH='configs/lstm.yaml'

python predict.py --config $OPTIONS_PATH
