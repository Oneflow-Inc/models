set -aux


OPTIONS_PATH='configs/lstm.yaml'

python preprocess.py --config $OPTIONS_PATH

echo "Data prepared !"
echo "Starting training ..."

python train.py --config $OPTIONS_PATH

echo "Train done!"
