# Pix2pix

## Introduction
Pix2Pix is a conditional image generation method, our code is inspired by [TensorFlow Tutorial](https://tensorflow.google.cn/tutorials/generative/pix2pix) and github repository[ pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## Installation
oneflow >= 0.4.0

## Prepare data
If you want to run train and test on small dataset (e.g., Facades, about 30.9M), running  the train.sh file automatically download this dataset in "./data/facades".

## Train
```
bash train.sh
```

## Download pretrained model
If you just want to run test, we provide a pretrained model. You can run the infer.sh file and the pretrained model will be downloaded automatically.

## Test
```
bash infer.sh
```
