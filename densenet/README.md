# DenseNet

## Train on [imagenette](https://github.com/fastai/imagenette) Dataset

### Prepare Traning Data

#### Download Ofrecord

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenette_ofrecord.tar.gz
tar zxf imagenette_ofrecord.tar.gz
```

### Run Oneflow Training script

```bash
bash train.sh
```


## Inference on Single Image

```bash
bash infer.sh
```

#### Model
We provide a converted pretrained model(from pytroch), you can get [here](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/densenet/densenet_121_oneflow_model.zip)
Also, you can use following steps to convert it on your own:

```sh
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/densenet/densenet121-a639ec97.pth
```

```python
import torch
import re
state_dict = torch.load("./densenet121-a639ec97.pth")
pattern = re.compile(
    r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

new_parameters = dict()
for key in list(state_dict.keys()):
    res = pattern.match(key)
    if res:
        new_key = res.group(1) + res.group(2)
        state_dict[new_key] = state_dict[key]
        del state_dict[key]

new_parameters = dict()
for key,value in state_dict.items():
    new_parameters[key] = value.detach().cpu().numpy()
densenet121_module.load_state_dict(new_parameters)
flow.save(densenet121_module.state_dict(), "densenet_121_oneflow_model")
print("model weight convert success!")
```