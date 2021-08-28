## Introduction
A simple and effective road semantic segmentation model, including detailed annotations, easy to use, very suitable for novices to get started.This repo is based on this [Python blog.](https://github.com/Yannnnnnnnnnnn/learnPyTorch/blob/master/road%20segmentation%20(camvid).ipynb)

## Installation

The code requires the following libraries:

- **numpy**   
- **oneflow**   0.5.0.dev20210801+cu102
- **cv2**   
- **matplotlib**   
- **albumentations**   


## Dataset
[https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/seg/FODDET/CamVid.rar](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/seg/FODDET/CamVid.rar)

## Trained Models
[https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/seg/FODDET/UNetmodel.rar](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/seg/FODDET/UNetmodel.rar)

## Code Function Introduction

- The training code `train.py` mainly contains data reading code, segmentation model code, and specific training code.

- The test code `test.py` mainly visualizes the segmentation effect, and the code content mainly includes the import of the trained model and the test process.You can download the trained model and use it directly.

## Training Scripts 

 ` bash train_oneflow.sh`

## Test Images

 ` bash test_oneflow.sh`

## Demonstration
The following shows some results of road image segmentation.


<img src="https://z3.ax1x.com/2021/08/26/hm34OA.png"/>

<img src="https://z3.ax1x.com/2021/08/26/hm3hyd.png"/>
