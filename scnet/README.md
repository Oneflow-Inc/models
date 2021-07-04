# Scnet

Scnet is a vehicle color classification network based on scloss, which can be easily combined with different backbone networks.

## Inference on Single Image

```bash
bash infer.sh
```


## Train on [hustcolor](http://cloud.eic.hust.edu.cn:8071/~pchen/color.rar) Dataset

### Prepare Traning Data
If you use the vehicle color recognition dataset for testing your recognition algorithm you should try and make your results comparable to the results of others. We suggest to choose half of the images in each category to train a model. And use the other half images to test the recognition algorithm.

#### Download Ofrecord

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenette_ofrecord.tar.gz
tar zxf imagenette_ofrecord.tar.gz
```

### Run Oneflow Training script
We have installed visdom to visualize the training model, and run the following program to enter http://localhost:8097/ get the training curve.

```
python -m visdom.server

```
```bash
bash train_oneflow.sh
```

### Performer of model
|         | val(Top1) |
| :-----: | :-----------------: |
| resnet  |        0.925        |
| scnet   |        0.947        |
