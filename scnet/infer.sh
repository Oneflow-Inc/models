set -aux

PRETRAIN_MODEL_PATH="scnet_acc_0.947254"
IMAGE_PATH="data/img_red.png"

python3 infer.py --model_path $PRETRAIN_MODEL_PATH --image_path $IMAGE_PATH
