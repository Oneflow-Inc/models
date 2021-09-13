# OneFlow-Models
**Models and examples implement with OneFlow(version >= 0.5.0).**

## Introduction
English | [简体中文](/README_zh-CN.md)

OneFlow-Models is an open source repo which contains official implementation of different models built on OneFlow. In each model, we provide at least two scripts `train.sh` and `infer.sh` for a quick start. For each model, we provide a detailed `README` to introduce the usage of this model.

## Features
- various models and pretrained weight
- easy use for beginners

## Official Models
**Quick Start**

- [demo](/demo/)

**Model List**
- [Computer Vision](/vision/)
  - [Image Classification](/vision/classification/image/)
  - [Video Classification](/vision/classification/video/)
  - [Object Detection](/vision/detection/)
  - [Semantic Segmentation](/vision/segmentation/)
  - [Generative Adversarial Networks](/vision/gan/)
  - [Neural Style Transform](/vision/style_transform/)
  - [Person Re-identification](/vision/reid/)
- [Natural Language Processing](/nlp/)
- [Audio](/audio/)
- [Deep Reinforcement Learning](/deep_reinforcement_learning/)
- [Quantization Aware Training](/quantization/)

## Installation and Environment setup
**Install Oneflow**

https://github.com/Oneflow-Inc/oneflow#install-with-pip-package

**Build custom ops from source**

In the root directory, run:
```bash
mkdir build
cd build
cmake ..
make -j$(nrpoc)
```
Example of using ops:
```bash
from ops import RoIAlign
pooler = RoIAlign(output_size=(14, 14), spatial_scale=2.0, sampling_ratio=2)
```

