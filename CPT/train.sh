set -aux

LR=1e-5
BATCH_SIZE=12
LOAD_DIR="."
SAVE_DIR="cpt_pretrain_afqmc"
NEPOCHS=30
CPT_PRETRAIN_DIR="cpt-base"
TASK="afqmc"

if [ ! -d $SAVE_DIR ]; then
    mkdir $SAVE_DIR
fi

if [ ! -d $CPT_PRETRAIN_DIR ]; then
    echo "Downloading cpt-base."
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/nlp/CPT/cpt-base.tar.gz
    tar -xzf cpt-base.tar.gz
fi

python train_flow.py \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --n_epochs $NEPOCHS \
    --model_load_dir $LOAD_DIR \
    --model_save_dir $SAVE_DIR \
    --pretrain_dir $CPT_PRETRAIN_DIR \
    --task $TASK \
    --cuda

