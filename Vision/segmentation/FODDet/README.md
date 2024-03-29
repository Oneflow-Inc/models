## Introduction
A simple and effective road semantic segmentation model, easy to use, very suitable for novices to get started.This repo is based on this [Python blog.](https://github.com/Yannnnnnnnnnnn/learnPyTorch/blob/master/road%20segmentation%20(camvid).ipynb)

## Installation

Install OneFlow:
- **oneflow**   0.5.0.dev20210801+cu102

Install other requirements
```
$ pip install -r requirements.txt
```    


## Dataset
Download required dataset from the following link:
[https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/seg/FODDET/CamVid.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/seg/FODDET/CamVid.zip)

## Trained Models
Download pretrained model from the following link:
[https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/seg/FODDET/UNetmodel.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/seg/FODDET/UNetmodel.zip)

## Code Function Introduction

- The training code `train.py` mainly contains data reading code, segmentation model code, and specific training code.

- The test code `test.py` mainly visualizes the segmentation effect, and the code content mainly includes the import of the trained model and the test process.You can download the trained model and use it directly.

## Training Scripts 
```
$ bash train/train.sh
```
## Test Images
```
$ bash test/test.sh
```
## Demonstration
The following shows some results of road image segmentation.


<img src="https://z3.ax1x.com/2021/08/26/hm34OA.png"/>

<img src="https://z3.ax1x.com/2021/08/26/hm3hyd.png"/>
