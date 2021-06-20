# ShuffleNet

This repo is based on: https://github.com/pytorch/vision/blob/master/torchvision/models/shufflenetv2.py

## Inference on Single Image

```bash
bash infer.sh
```

## Train on imagenette Dataset

### Prepare Traning Data

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenette_ofrecord.tar.gz
tar zxf imagenette_ofrecord.tar.gz
```

### Run Oneflow Training Script

```bash
bash train.sh
```