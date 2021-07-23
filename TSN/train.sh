set -aux

PRETRAIN_MODEL_PATH="resnet50_imagenet_pretrain_model"
DATASET_PATH="./data"
SAVE_PATH="./save_path"

if [ ! -d "$DATASET_PATH" ]; then
    echo "Error! The training data set is empty! Please refer to (https://github.com/open-mmlab/mmaction/tree/master/data_tools/kinetics400) for data prepration."
    exit
fi

if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/action_recognition/resnet50_imagenet_pretrain_model.zip
    unzip resnet50_imagenet_pretrain_model.zip
fi

if [ ! -d "$SAVE_PATH" ]; then
    mkdir save_path
fi

python3 train_recognizer.py --pretrained $PRETRAIN_MODEL_PATH --data_dir $DATASET_PATH --save_checkpoint_path $SAVE_PATH
