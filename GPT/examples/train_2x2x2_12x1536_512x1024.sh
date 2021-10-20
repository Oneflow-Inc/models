# set -aux

DEVICE_NUM_PER_NODE=4
NUM_NODES=${1:-"1"}
NODE_RANK=${2:-"0"}
MASTER_ADDR=${3:-"127.0.0.1"}

export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED
export NCCL_LAUNCH_MODE=GROUP
echo NCCL_LAUNCH_MODE=$NCCL_LAUNCH_MODE
export NCCL_DEBUG=INFO
export ONEFLOW_DEBUG_MODE=True
export ONEFLOW_COMM_NET_IB_ENABLE=True

DATASET=/dataset/gpt/gpt_sample_dataset_text_document
SEQ_LEN=1024
LAYER_NUM=12
HIDDEN_SIZE=1536
HEAD_NUM=16
MBZ=8
GBZ=512
ACC_STEPS=32
TMP=2
PMP=2
TRAIN_ITER=5
LOG_INTERVAL=1

SRC_DIR=$(realpath $(dirname $0)/..)

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
        --micro-batch-size $MBZ \
        --global-batch-size $GBZ \
        --num-accumulation-steps $ACC_STEPS \
        --tensor-model-parallel-size $TMP \
        --pipeline-model-parallel-size $PMP \
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
