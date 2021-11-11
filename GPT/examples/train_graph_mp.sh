#! /bin/bash

export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED
export ONEFLOW_DEBUG_MODE=True

DATASET=/dataset/gpt/gpt_sample_dataset_text_document
SEQ_LEN=1024
LAYER_NUM=12
HIDDEN_SIZE=768
HEAD_NUM=12

MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=32
TENSOR_MODEL_PARALLEL_SIZE=4
PIPELINE_MODEL_PARALLEL_SIZE=1

ACC_STEPS=1
TRAIN_ITER=300
LOG_INTERVAL=1

SRC_DIR=$(realpath $(dirname "$0")/..)

DEVICE_NUM_PER_NODE=2
MASTER_ADDR=127.0.0.1
NUM_NODES=1
NODE_RANK=0

# gdb --args \
python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
$SRC_DIR/oneflow_gpt/train.py \
    --dataset $DATASET \
    --split 949,50,1 \
    --vocab-size 50257 \
    --seq-length $SEQ_LEN \
    --num-layers $LAYER_NUM \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $HEAD_NUM \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE \
    --pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE \
    --checkpoint-activations \
    --num-gpus-per-node $DEVICE_NUM_PER_NODE \
    --num-nodes $NUM_NODES \
    --optimizer adamw \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --initial-loss-scale 4294967296 \
    --learning-rate 0.00015 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --lr-decay-iters 320000 \
    --lr-warmup-fraction 0.01 \
    --no-scale-tril-softmax-dropout-fusion \
    --fp16 \
    --graph \
    --train-iters $TRAIN_ITER \
    --log-interval $LOG_INTERVAL \
    # --zero_stage_1 \
