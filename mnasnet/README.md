# MNASNet
Training MNASNet on [imagenette](https://github.com/fastai/imagenette) Dataset using OneFlow

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
We only provide pretrained `mnasnet0_5` and `mnasnet1_0` weight
```bash
mkdir weight
```

- for `mnasnet0_5`
```bash
cd weight
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/MNASNet/mnasnet0_5.zip
unzip mnasnet0_5.zip
```

- for `mnasnet1_0`
```bash
cd weight
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/MNASNet/mnasnet1_0.zip
unzip mnasnet1_0.zip
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
