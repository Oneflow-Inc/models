# Posenet

Posenet is a backbone network, we use it to classify face pose.

## Inference on Single Image

```bash
bash infer.sh
```


## Train on face pose dataset Dataset
The face pose dataset can be downloaded from [BaiduNetdis](https://pan.baidu.com/s/1KbrMUrUIS_cCzpDgdgjMRQ) code: o9tg .

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
| Posenet  |        0.906        |
