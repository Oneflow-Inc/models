# Mobilenetv3

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

#### Util
pytorch pretrained module to oneflow
```python
wget https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv3_1.0-f2a8633.pth.tar?dl=1
parameters = torch.load("./mobilenetv3_1.0-f2a8633.pth.tar")
for key,value in parameters.items():
     val = value.detach().cpu().numpy()
     parameters[key] = val
     print("key:", key, "value.shape", val.shape)
mobilenetv3_module.load_state_dict(parameters)
flow.save(mobilenetv3_module.state_dict(), "mobilenetv3_oneflow_model")
```