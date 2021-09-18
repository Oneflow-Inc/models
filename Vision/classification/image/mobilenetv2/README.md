# MobilenetV2

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
We provide a converted pretrained model(from pytroch), you can get [here](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/mobilenetv2/mobilenetv2_oneflow_model.zip)
Also, you can use following steps to convert it on your own:

```sh
wget https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
```

```python
parameters = torch.load("./mobilenet_v2-b0353104.pth")

new_parameters = dict()
for key,value in parameters.items():
     if "num_batches_tracked" not in key:
          val = value.detach().cpu().numpy()
          new_parameters[key] = val

mobilenetv2_module.load_state_dict(new_parameters)
flow.save(mobilenetv2_module.state_dict(), "mobilenetv2_oneflow_model")
```