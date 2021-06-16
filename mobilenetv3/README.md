# Mobilenetv3

## Train on [imagenette](https://github.com/fastai/imagenette) Dataset

### Prepare Traning Data And Pretrain Models

#### Download Ofrecord

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenette_ofrecord.tar.gz
tar zxf imagenette_ofrecord.tar.gz
```

### Download Pretrain Models

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/mobilenetv3/mobilenetv3_oneflow_model.tar.gz
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

convert pytorch pretrained model to oneflow

```python
import torch
parameters = torch.load("mobilenet_v3_small-047dcff4.pth")
new_parameters = dict()
for key,value in parameters.items():
     if "num_batches_tracked" not in key:
          val = value.detach().cpu().numpy()
          new_parameters[key] = val
mobilenetv3_module.load_state_dict(new_parameters)
flow.save(mobilenetv3_module.state_dict(), "mobilenetv3_oneflow_model")
```
