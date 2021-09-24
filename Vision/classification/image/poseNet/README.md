# Posenet

Posenet is a backbone network, we use it to classify face pose.

## Train on face pose dataset Dataset
The face pose dataset can be downloaded from [BaiduNetdis](https://pan.baidu.com/s/1KbrMUrUIS_cCzpDgdgjMRQ) code: o9tg .

#### Download Ofrecord

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/pose/pose_dataset.zip
unzip pose_dataset.zip
```

### Run Oneflow Training script

```
pip3 install -r requirements.txt --user
```

```bash
bash train_oneflow.sh
```
## Inference on Single Image

```bash
bash infer.sh
```

### Accuracy
|         | val(Top1) |
| :-----: | :-----------------: |
| Posenet  |        0.906        |
