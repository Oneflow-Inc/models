# GhostNet

This repo is based on: https://github.com/huawei-noah/CV-Backbones/tree/master/ghostnet_pytorch/ghostnet.py


#### Download Ofrecord

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenette_ofrecord.tar.gz
tar zxf imagenette_ofrecord.tar.gz
```

### Run Oneflow Training script

```bash
bash train_oneflow.sh
```

## Inference on Single Image

```bash
bash infer.sh
```

## Compare with pytorch_ghostnet

![Compare of loss](https://github.com/Oneflow-Inc/models/ghostnet/utils/ghostnet_compare.png)

