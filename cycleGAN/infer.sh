TRAIN_DATASET="horse2zebra"

TESTA_DIR="./datasets/${TRAIN_DATASET}/testA/"
TESTB_DIR="./datasets/${TRAIN_DATASET}/testB/"

NETG_A_DIR="htz_model_oneflow/"
NETG_B_DIR="zth_model_oneflow/"

SAVE_A_DIR="fake_B/"
SAVE_B_DIR="fake_A/"

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
    # --checkpoint_load_dir $CHECKPOINT_LOAD_DIR

