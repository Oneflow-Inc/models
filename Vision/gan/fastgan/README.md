
Modified from https://github.com/odegeasslbc/FastGAN-pytorch

## Training
- 多卡
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m oneflow.distributed.launch --nproc_per_node 8 --master_port 12345 train.py
```

- 单卡
```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py
```

## Inference
- 打开对应的保存文件的位置,执行下面的命令
```bash
CUDA_VISIBLE_DEVICES=1 python3 eval.py --start_iter 1 --end_iter 2 --im_size 256 --n_sample 100 --artifacts /home/shikaijie/models/Vision/gan/fastgan/train_results/test 
```

## FID
```bash
python3 fid.py --size 256 --path_b ../train_results/test --path_a /home/shikaijie/models/Vision/gan/fastgan/100-shot-panda --iter 1 --end 5
```

## TODO
1. 写好脚本，方便FID的计算
