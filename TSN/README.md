# TSN

This repo provides one method for Person Action Recognition task.

## Get started

```bash
pip3 install -r requirements.txt
```

## Data preparation

We train and evaluate our models on Kinetics400 dataset. Please refer to (https://github.com/open-mmlab/mmaction/tree/master/data_tools/kinetics400) for data prepration.

Download the mini_dataset to evaluate the model.

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/TSN/kinetics400.zip
unzip kinetics400.zip
```

## Training

```bash
bash train.sh
```
Our model is trained using ResNet50 as backbones

## Evaluating

```bash
bash infer.sh
```

## Accuracy

            oneflow     torch
top1 acc    0.6373      0.6373
top5 acc    0.6785      0.6785
