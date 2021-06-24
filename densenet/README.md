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
We provide a converted pretrained model(from pytroch), you can get [here](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/densenet/densenet121_oneflow_model.zip)
Also, you can use following steps to convert it on your own:

```sh
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/densenet/densenet121-a639ec97.pth 
```

```python
# import re
pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
parameters = torch.load("./densenet121-a639ec97.pth")

new_parameters = dict()
for key,value in parameters.items():
    res = pattern.match(key)
    if res:
        new_key = res.group(1) + res.group(2)
        val = value.detach().cpu().numpy()
        new_parameters[new_key] = val
    else:
        val = value.detach().cpu().numpy()
        new_parameters[key] = val

densenet_121_module.load_state_dict(new_parameters)
flow.save(densenet_121_module.state_dict(), "densenet_121_oneflow_model")
```