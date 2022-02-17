```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m oneflow.distributed.launch --nproc_per_node 8 --master_port 12345 pretrain.py --data_path /dataset/extract --batch_size 4
```