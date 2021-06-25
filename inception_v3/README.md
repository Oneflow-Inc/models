# InceptionV3

## Train on [imagenette](https://github.com/fastai/imagenette) Dataset

### Prepare Traning Data And Pretrain Models

#### Download Ofrecord

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenette_ofrecord.tar.gz
tar zxf imagenette_ofrecord.tar.gz
```

### Download Pretrain Models

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/inceptionv3/inceptionv3_oneflow_model.tar.gz
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

```sh
wget https://download.pytorch.org/models/inceptionv3-owt-7be5be79.pth
```

```python
import torch
import oneflow as flow 
from models.inceptionv3 import inceptionv3

parameters = torch.load("inceptionv3-owt-7be5be79.pth")
new_parameters = dict()
for key,value in parameters.items():
     if "num_batches_tracked" not in key:
          val = value.detach().cpu().numpy()
          new_parameters[key] = val

flow.env.init()
flow.enable_eager_execution()

inceptionv3_module = inceptionv3()
inceptionv3_module.load_state_dict(new_parameters)
flow.save(inceptionv3_module.state_dict(), "inceptionv3_oneflow_model")
```
