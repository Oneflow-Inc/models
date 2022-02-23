
## pretrain
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m oneflow.distributed.launch --nproc_per_node 8 --master_port 12345 pretrain.py --data_path /dataset/extract --batch_size 16
```

## linprobe
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m oneflow.distributed.launch --nproc_per_node 8 --master_port 12355 linprobe.py --data_path /dataset/extract --finetune ./output_dir/checkpoint-180.pth --output_dir ./output_dir_linprobe --log_dir ./output_dir_linprobe   --batch_size 128
```