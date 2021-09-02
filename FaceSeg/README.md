# Face segmentation

This repo is about face segmentation based on LinkNet34, our work is inspired by this PyTorch [implementaion](https://github.com/JiaojiaoYe1994/face-segmentation).

## Environment
| Spec                        |                                                             |
|-----------------------------|-------------------------------------------------------------|
| Operating System            | Ubuntu 18.04                                        |
| GPU                         | Nvidia A100-SXM4-40GB                          |
| CUDA Version                | 11.0                                                   |
| Driver Version              | 460.73.01                                             |
| Oneflow Version 	          | branch: master, commit_id: 90d3277a098f483d0a0e68621b7c8fb2497a6fc2 |


### Prerequisites

```
pip3 install -r requirements.txt
```


### Download Pretrain Models

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/seg/linknet_oneflow_model.zip
```


## Inference to segment face for webcam

```bash
bash infer.sh
```


## Train 

### Data preparation

We combine several dataset and preprocess them, for training and evaluating our 
face segmentation model. We provide the final compressed dataset, please download dataset from 
[here](https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/faceseg_data.zip) , unzip it and put it in `./`

### Backbone 
Please download pretrained ResNet backbone from [pretrained_resnet_oneflow_model](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/seg/pretrained_resnet_oneflow_model.zip)
, unzip it and put it in `./`

### Run Oneflow Training script

Hyperparameters can be customized, the provided model is trained by run

```bash
bash train.sh
```

## Results
### Accuracy
 
We compare our model with baseline implemented using PyTorch, our model on OneFlow performs even little bit better on train dataset than baseline. The result  is shown in the following,

 |         | Train IoU (%) | Test IoU (%) |
| :------ | :-----------: | -----------: |
| PyTorch |    92.815     |       92.689 |
| OneFlow |    95.006     |     94.425| |
 
### Demonstration
The following displays some segmented results on grabbed videos.

<img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/seg/demo.gif"/>
<img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/seg/demo2.gif"/>
