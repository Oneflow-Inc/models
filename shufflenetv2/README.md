# ShuffleNet

This repo is based on: https://github.com/pytorch/vision/blob/master/torchvision/models/shufflenetv2.py

## Inference on Single Image

```bash
bash infer.sh
```

## Train on imagenette Dataset

### Prepare Traning Data

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenette_ofrecord.tar.gz
tar zxf imagenette_ofrecord.tar.gz
```

### Run Oneflow Training Script

```bash
bash train.sh
```

## Convert Pretrained Model From PyTorch To OneFlow

Download PyTorch pretrained model first:

```shell
wget https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth
wget https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth
```

And then run the code below:

```python
import numpy as np
import torch
import oneflow.experimental as flow
from models.shufflenetv2 import shufflenetv2_x0dot5, shufflenetv2_x1

flow.enable_eager_execution()
model_dict = {
    "shufflenetv2_x0.5": {"model": shufflenetv2_x0dot5, "url":"https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",
    "file": "shufflenetv2_x0.5-f707e7126e.pth"},
    "shufflenetv2_x1.0": {"model": shufflenetv2_x1, "url":"https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
    "file": "shufflenetv2_x1-5666bf0f80.pth"},
}

for k, v in model_dict.items():
    m = v["model"]()

    parameters = torch.load(v["file"])
    for key, value in parameters.items():
        val = value.detach().cpu().numpy()
        parameters[key] = val
        print("key:", key, "value.shape", val.shape)

    m.load_state_dict(parameters)
    flow.save(m.state_dict(), "oneflow_" + k)
```

The converted OneFlow model will be saved to `oneflow_shufflenetv2_xxx` directory.