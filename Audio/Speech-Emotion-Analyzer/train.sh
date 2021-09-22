set -aux


OPTIONS_PATH='configs/lstm.yaml'

#python preprocess.py --config $OPTIONS_PATH

echo "Data prepared !"
echo "Starting training ..."

CUDA_VISIBLE_DEVICES=2 python train.py --config $OPTIONS_PATH

echo "Train done!"
