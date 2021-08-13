# set -aux

TOTAL_DEVICE_NUM=2
MASTER_ADDR=127.0.0.1
NUM_NODES=1
NODE_RANK=0

python3 -m oneflow.distributed.launch \
    --nproc_per_node $TOTAL_DEVICE_NUM \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    graph/inplace_lazy.py
