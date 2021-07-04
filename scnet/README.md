# Scnet

Scnet is a vehicle color classification network based on scloss, which can be easily combined with different backbone networks.

## Inference on Single Image

```bash
bash infer.sh
```


## Train on [hustcolor](http://cloud.eic.hust.edu.cn:8071/~pchen/color.rar) Dataset

### Prepare Traning Data

#### Download Ofrecord

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenette_ofrecord.tar.gz
tar zxf imagenette_ofrecord.tar.gz
```

### Run Oneflow Training script

```bash
bash train_oneflow.sh
```

### Performer of model
|         | val(Top1) |
| :-----: | :-----------------: |
| resnet  |        0.925        |
| scnet   |        0.947        |