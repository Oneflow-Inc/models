# Mobilenet

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
wget https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenet_1.0-f2a8633.pth.tar?dl=1
parameters = torch.load("./mobilenet_1.0-f2a8633.pth.tar")
for key,value in parameters.items():
     val = value.detach().cpu().numpy()
     parameters[key] = val
     print("key:", key, "value.shape", val.shape)
mobilenet_module.load_state_dict(parameters)
flow.save(mobilenet_module.state_dict(), "mobilenet_oneflow_model")
```