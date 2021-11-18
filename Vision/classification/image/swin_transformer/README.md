# SwinTransformer

## Train on [imagenette](https://github.com/fastai/imagenette) Dataset

### Prepare Traning Data And Pretrain Models

#### Download Ofrecord

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenette_ofrecord.tar.gz
tar zxf imagenette_ofrecord.tar.gz
```

### Download Pretrain Models

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/swin_transformer/swin_tiny_patch4_window7_224_oneflow_model.tar.gz
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

Here we use swin_tiny_patch4_window7_224 as an example. 

```python
import torch
import oneflow as flow 
from models.swin_transformer import create_swin_transformer
import numpy as np 


def convert_torch_to_flow(model, torch_weight_path, save_path):
     parameters = torch.load(torch_weight_path)
     new_parameters = dict()
     for key, value in parameters["model"].items():
          if "relative_position_index" in key: 
               val = value.detach().cpu().numpy() # here index is int64 type, do not turn as float32!
               new_parameters[key] = val
          elif "num_batches_tracked" not in key:
               val = value.detach().cpu().numpy().astype(np.float32)
               new_parameters[key] = val

     model.load_state_dict(new_parameters)
     flow.save(model.state_dict(), save_path)
     print("successfully save model to %s" % (save_path))


swin_transformer = create_swin_transformer()
convert_torch_to_flow(swin_transformer, "swin_tiny_patch4_window7_224.pth",  "swin_tiny_patch4_window7_224_oneflow_model")
```
