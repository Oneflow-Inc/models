# Resnet50
Training Resnet50 on [imagenette](https://github.com/fastai/imagenette) Dataset using OneFlow

## Usage
### 0. Requirements
Experiment environment:
- oneflow
- tqdm
- tensorboardX (optional)

### 1. Prepare Traning Data And Pretrain Models
#### Download Ofrecord
```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenette_ofrecord.tar.gz
tar zxf imagenette_ofrecord.tar.gz
```

#### Download Pretrain Models

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/resnet50_imagenet_pretrain_model.tar.gz
tar zxf resnet50_imagenet_pretrain_model.tar.gz
```

### 2. Run Oneflow Training Script
#### Eager Training Scripts
```bash
bash eager/train.sh
```

#### Graph Training Scripts
```bash
bash graph/train.sh
```


### 3. Inference on Single Image
#### Eager Inference
```bash
bash eager/infer.sh
```

#### Graph Inference
```bash
bash graph/infer.sh
```

## Util
### 1. Model Compare
Compare Resnet50 model on different training mode (Graph / Eager)
```bash
bash check/check.sh
```
Compare results will be saved to `./results/check_report.txt`
