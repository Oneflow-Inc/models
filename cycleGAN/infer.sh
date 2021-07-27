TRAIN_DATASET="summer2winter_yosemite" 
#choose between apple2orange, horse2zebra, summer2winter_yosemite. Note that you need to download the corresponding dataset first.

TESTA_DIR="./datasets/${TRAIN_DATASET}/testA/"
TESTB_DIR="./datasets/${TRAIN_DATASET}/testB/"

if [ ! -d "cycleGAN/" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/gan/pretrained.tar.gz
    tar -zxvf pretrained.tar.gz
fi

NETG_A_DIR="cycleGAN/${TRAIN_DATASET}/"
NETG_B_DIR="cycleGAN/${TRAIN_DATASET}_reverse/"

SAVE_A_DIR="${TRAIN_DATASET}_images/"
SAVE_B_DIR="${TRAIN_DATASET}_reverse_images/"

if [ ! -d $SAVE_A_DIR ]; then
    mkdir $SAVE_A_DIR
fi

if [ ! -d $SAVE_B_DIR ]; then
    mkdir $SAVE_B_DIR
fi

CUDA_VISIBLE_DEVICES=1 python3 infer.py \
    --datasetA_path $TESTA_DIR \
    --datasetB_path $TESTB_DIR \
    --netG_A_dir $NETG_A_DIR \
    --netG_B_dir $NETG_B_DIR \
    --fake_B_save_dir $SAVE_A_DIR \
    --fake_A_save_dir $SAVE_B_DIR \

