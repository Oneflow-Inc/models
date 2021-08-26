## Introduction
A simple and effective road semantic segmentation model, including detailed annotations, easy to use, very suitable for novices to get started.This repo is based on this [Python blog.](https://github.com/Yannnnnnnnnnnn/learnPyTorch/blob/master/road%20segmentation%20(camvid).ipynb)

## Installation

The code requires the following libraries:

- numpy
- oneflow
- cv2
- matplotlib
- albumentations

## File structure

    #dataset folder
    data  /
          /data_file

    #save models folder      
    result/
          /save_models_file

    #training code      
    train.py

    #test file      
    test  /
          /test.py
          /dataloader.py
          /UNet.py
          /visualize.py



## Dataset
[https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/seg/FODDET/CamVid.rar](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/seg/FODDET/CamVid.rar)

## Trained Models
[https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/seg/FODDET/UNetmodel.rar](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/seg/FODDET/UNetmodel.rar)

## Code function introduction

- The training code `train.py` mainly contains data reading code, segmentation model code, and specific training code.

- The test code `test.py` mainly visualizes the segmentation effect, and the code content mainly includes the import of the trained model and the test process.You can download the trained model and use it directly.

## Training script

 ` sh train_oneflow.sh`

## steps
1. prepare the environment
2. clone code
3. Prepare the data set
4. Set the data path
5. Set training parameters
6. Start training


## Demonstration
The following shows some results of road image segmentation.


<img src="https://z3.ax1x.com/2021/08/26/hm34OA.png"/>

<img src="https://z3.ax1x.com/2021/08/26/hm3hyd.png"/>
