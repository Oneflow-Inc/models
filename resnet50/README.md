# Resnet50

This repo is based on: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

## Inference on Single Image

```bash
bash infer.sh
```

## Compare the Training Efficience Between Oneflow and Pytorch

```python3
python3 compare_oneflow_and_pytorch_resnet50_speed.py
```

## Train on [imagenette](https://github.com/fastai/imagenette) Dataset

### Prepare Traning Data

#### Download Ofrecord

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenette_ofrecord.tar.gz
tar zxf imagenette_ofrecord.tar.gz
```

### Run Oneflow Training script

```bash
bash train_oneflow.sh
```

