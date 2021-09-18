# AlexNet
Training Alexnet on [imagenette](https://github.com/fastai/imagenette) Dataset using OneFlow

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
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/alexnet/alexnet_oneflow_model.tar.gz
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
Compare Alexnet model on different training mode (Graph / Eager)
```bash
bash check/check.sh
```
Compare results will be saved to `results/check_info`

Compare Results Picture
```bash
bash check/draw.sh
```
The pictures will be saved to `results/pictures`

### 2. Convert Pretrained Model Weight
convert pytorch pretrained model to oneflow pretrained model

```sh
wget https://download.pytorch.org/models/alexnet-owt-7be5be79.pth
```

```python
import torch
import oneflow as flow 
from models.alexnet import alexnet

parameters = torch.load("alexnet-owt-7be5be79.pth")
new_parameters = dict()
for key,value in parameters.items():
     if "num_batches_tracked" not in key:
          val = value.detach().cpu().numpy()
          new_parameters[key] = val




alexnet_module = alexnet()
alexnet_module.load_state_dict(new_parameters)
flow.save(alexnet_module.state_dict(), "alexnet_oneflow_model")
```
