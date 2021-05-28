# Resnet50

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

#### Download Raw Dataset

```
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenette2.tgz
tar zxf imagenette2.tgz
```

### Run Oneflow Training script

```bash
python3 train_resnet50_oneflow.py
```

### Run Pytorch Training script

```bash
python3 train_resnet50_pytorch.py
```
