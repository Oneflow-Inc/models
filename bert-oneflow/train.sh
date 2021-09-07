#!/bin/bash
set -aux

_DEVICE_NUM_PER_NODE=1
_MASTER_ADDR=127.0.0.1
_NUM_NODES=1
_NODE_RANK=0

# export PYTHONUNBUFFERED=1
# echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED
# export NCCL_LAUNCH_MODE=PARALLEL
# echo NCCL_LAUNCH_MODE=$NCCL_LAUNCH_MODE

OFRECORD_PATH="/dataset/bert_regression_test/0"
# if [ ! -d "$OFRECORD_PATH" ]; then
#     wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/wiki_ofrecord_seq_len_128_example.tgz
#     tar zxf wiki_ofrecord_seq_len_128_example.tgz
# fi

CHECKPOINT_PATH="checkpoints"
if [ ! -d "$CHECKPOINT_PATH" ]; then
    mkdir $CHECKPOINT_PATH
fi

LEARNING_RATE=0.02
EPOCH=1
TRAIN_BATCH_SIZE=32
VAL_BATCH_SIZE=32

export CUDA_VISIBLE_DEVICES=2
python3 -m oneflow.distributed.launch \
    --nproc_per_node $_DEVICE_NUM_PER_NODE \
    --nnodes $_NUM_NODES \
    --node_rank $_NODE_RANK \
    --master_addr $_MASTER_ADDR \
    run_pretraining.py \
        --ofrecord_path $OFRECORD_PATH \
        --checkpoint_path $CHECKPOINT_PATH \
        --lr $LEARNING_RATE \
        --epochs $EPOCH \
        --train_batch_size $TRAIN_BATCH_SIZE \
        --val_batch_size $VAL_BATCH_SIZE \
        --seq_length=128 \
        --max_predictions_per_seq=20 \
        --num_hidden_layers=12 \
        --num_attention_heads=12 \
        --max_position_embeddings=512 \
        --type_vocab_size=2 \
        --vocab_size=30522 \
        --attention_probs_dropout_prob=0.1 \
        --hidden_dropout_prob=0.1 \
        --hidden_size_per_head=64 \
        --use_fp16=True \
        --use_consistent=True \
        --metric-local=True 