# InceptionV3

## Train on [imagenette](https://github.com/fastai/imagenette) Dataset

### Prepare Traning Data And Pretrain Models

#### Download Ofrecord

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenette_ofrecord.tar.gz
tar zxf imagenette_ofrecord.tar.gz
```

#### Download Pretrain Models

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/inceptionv3/inceptionv3_oneflow_model.tar.gz
```

### Eager Training script

```bash
bash eager/train.sh
```


### Eager Inference on Single Image

```bash
bash eager/infer.sh
```

### Graph Training script
```bash
bash graph/train_ddp.sh
```
- `DEVICE_NUM_PER_NODE`: to set the training gpu nums.

### Graph Inference on Single Image
```bash
bash graph/infer.sh
```

#### Util

convert pytorch pretrained model to oneflow pretrained model

```sh
wget https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth
```

```python
import torch
import oneflow as flow 
from models.inceptionv3 import inception_v3

parameters = torch.load("inception_v3_google-0cc3c7bd.pth")
new_parameters = dict()
for key,value in parameters.items():
     if "num_batches_tracked" not in key:
          val = value.detach().cpu().numpy()
          new_parameters[key] = val




inceptionv3_module = inception_v3()
inceptionv3_module.load_state_dict(new_parameters)
flow.save(inceptionv3_module.state_dict(), "inceptionv3_oneflow_model")
```
