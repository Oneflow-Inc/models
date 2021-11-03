

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

declare -a arr=("data" "row" "col")

## now loop through the above array
for fc1 in "${arr[@]}"
do
    # or do whatever with individual element of the array
    for fc2 in "${arr[@]}"
    do
        for fc3 in "${arr[@]}"
        do
            echo "start training with $fc1 $fc2 $fc3"
            python3 -m oneflow.distributed.launch --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 train.py --save ./graph_distributed_fp32_checkpoints --ofrecord-part-num 256 --num-devices-per-node 4 --num-epochs 1 --use-gpu-decode --scale-grad --graph --synthetic-data \
            --num-classes 1000 \
            --train-batch-size 16 \
            --model-name vgg \
            --parallel-way $fc1 $fc2 $fc3 \
            --write-file "vgg_throughput/fc1_$fc1.fc2_$fc2.fc3_$fc3.txt"
        done
    done
done

