set -aux


DATA_PATH="data"
if [ ! -d "$DATA_PATH" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/seq2seq_data.tar.gz
    tar -xzvf seq2seq_data.tar.gz
fi

DEVICE='cuda'
SAVE_ENCODER_CHECKPOINT_PATH="./saving_model_oneflow/encoder/"
SAVE_DECODER_CHECKPOINT_PATH="./saving_model_oneflow/decoder/"
HIDDDEN_SIZE=256
N_ITERS=75000
LR=0.01
DROP=0.1


python3 train_oneflow.py \
    --device $DEVICE \
    --save_encoder_checkpoint_path $SAVE_ENCODER_CHECKPOINT_PATH \
    --save_decoder_checkpoint_path $SAVE_DECODER_CHECKPOINT_PATH \
    --hidden_size $HIDDDEN_SIZE \
    --n_iters $N_ITERS \
    --lr $LR \
    --drop $DROP 