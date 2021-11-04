unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

declare -a arr=("data" "row" "col")
declare -a batch_size=(16 32 64)
declare -a num_classes=(1024 4096 16384 32768 65536 131072)

## now loop through the above array
for bs in "${batch_size[@]}"
do
    # or do whatever with individual element of the array
    for cls in "${num_classes[@]}"
    do
        for fc in "${arr[@]}"
        do
            echo "start training with $bs $cls $fc"
            python3 -m oneflow.distributed.launch --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 train.py --save ./graph_distributed_fp32_checkpoints --ofrecord-part-num 256 --num-devices-per-node 4 --num-epochs 1 --use-gpu-decode --scale-grad --graph --synthetic-data \
            --num-classes $cls \
            --train-batch-size $bs \
            --model-name resnet \
            --parallel-way $fc \
            --write-file "resnet_throughput/bsz_$bs.cls_$cls.fc_$fc.txt" 2>&1 | tee "resnet_log/bsz_$bs.cls_$cls.fc_$fc.txt"
        done
    done
done

    

