
export PYTHONPATH=/workspace/auto_parallel/oneflow/python:$PYTHONPATH

python3 -m oneflow.distributed.launch --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 train.py --save ./graph_distributed_fp32_checkpoints --ofrecord-part-num 256 --num-devices-per-node 4 --num-epochs 1 --use-gpu-decode --scale-grad --graph --synthetic-data --num-classes 4096 --train-batch-size 16 --parallel-way data data data --model-name vgg --write-file test