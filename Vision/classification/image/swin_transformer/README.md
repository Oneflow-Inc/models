# RepVGG

## Train on [imagenette](https://github.com/fastai/imagenette) Dataset

### Prepare Traning Data And Pretrain Models

#### Download Ofrecord

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenette_ofrecord.tar.gz
tar zxf imagenette_ofrecord.tar.gz
```

### Download Pretrain Models

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/repvgg/repvggA0_oneflow_model.tar.gz
```

### Run Oneflow Training script

```bash
bash train.sh
```


## Inference on Single Image

```bash
bash infer.sh
```

#### Util

convert pytorch pretrained model to oneflow pretrained model

you can download official pretrained model from here: 

Google Drive: https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq?usp=sharing

Baidu Cloud: https://pan.baidu.com/s/1nCsZlMynnJwbUBKn0ch7dQ

Here we use RepVGGA0 as an example. 

```python
import torch
import oneflow as flow 
from models.repvgg import create_RepVGG_A0

parameters = torch.load("RepVGG-A0-train.pth")
new_parameters = dict()
for key,value in parameters.items():
     if "num_batches_tracked" not in key:
          val = value.detach().cpu().numpy()
          new_parameters[key] = val




repVGGA0 = create_RepVGG_A0()
repVGGA0.load_state_dict(new_parameters)
flow.save(repVGGA0.state_dict(), "repvggA0_oneflow_model")
```
