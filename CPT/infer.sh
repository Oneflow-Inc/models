set -aux

LOAD_DIR="cpt_pretrain_afqmc"
CPT_PRETRAIN_DIR='/cpt-base'
TASK="afqmc"
TEXT1="订单退款成功了，为什么仍然要还花呗"
TEXT2="退款到了花呗，为什么还要还款"

if [ ! -d $LOAD_DIR ]; then
    echo "Please train CPT first."
    exit
fi

if [ ! -d $CPT_PRETRAIN_DIR ]; then
    echo "Downloading cpt-base."
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/nlp/CPT/cpt-base.tar.gz
    tar -xzf cpt-base.tar.gz
fi

python infer_flow.py \
    --model_load_dir $LOAD_DIR \
    --pretrain_dir $CPT_PRETRAIN_DIR \
    --text1 $TEXT1 \
    --text2 $TEXT2 \
    --task $TASK \
    --cuda

