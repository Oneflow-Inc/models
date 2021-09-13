# OneFlow-Models
**Models and examples implement with OneFlow(version >= 0.5.0).**

## Introduction
**English** | [简体中文](/README_zh-CN.md)

OneFlow-Models is an open source repo which contains official implementation of different models built on OneFlow. In each model, we provide at least two scripts `train.sh` and `infer.sh` for a quick start. For each model, we provide a detailed `README` to introduce the usage of this model.

## Features
- various models and pretrained weight
- easy use for beginners

## Official Models
**Quick Start**

- [demo](Demo)

**Model List**

- Computer Vision
  - Image Classification
    * [Lenet](Demo/quick_start_demo_lenet)
    * [Alexnet](Vision/classification/image/alexnet)
    * [VGG16/19](Vision/classification/image/vgg)
    * [Resnet50](Vision/classification/image/resnet50)
    * [InceptionV3](Vision/classification/image/inception_v3)
    * [Densenet](Vision/classification/image/densenet)
    * [Resnext50_32x4d](Vision/classification/image/resnext50_32x4d)
    * [Shufflenetv2](Vision/classification/image/shufflenetv2)
    * [MobilenetV2](Vision/classification/image/mobilenetv2)
    * [mobilenetv3](Vision/classification/image/mobilenetv3)
    * [Ghostnet](Vision/classification/image/ghostnet)
    * [Repvgg](Vision/classification/image/repvgg)
    * [DLA](Vision/classification/image/DLA)
    * [PoseNet](Vision/classification/image/poseNet)
    * [Scnet](Vision/classification/image/scnet)
    * [Mnasnet](Vision/classification/image/mnasnet)
  - Video Classification
    * [TSN](Vision/classification/video/TSN)
  - Object Detection
    * [CSRNet](Vision/detection/CSRNet)
  - Semantic Segmentation
    * [FODDet](Vision/segmentation/FODDet)
    * [FaceSeg](Vision/segmentation/FaceSeg)
  - Generative Adversarial Networks
    * [DCGAN](Vision/gan/DCGAN)
    * [SRGAN](Vision/gan/SRGAN)
    * [Pix2Pix](Vision/gan/pix2pix)
    * [CycleGAN](Vision/gan/cycleGAN)
  - Neural Style Transform
    * [FastNeuralStyle](Vision/style_transform/fast_neural_style)
  - Person Re-identification
    * [bot](Vision/reid/bot)
- Natural Language Processing
  * [RNN](NLP/rnn)
  * [Seq2Seq](NLP/seq2seq)
  * [LSTMText](NLP/LSTMText)
  * [TextCNN](NLP/TextCNN)
  * [Transformer](NLP/Transformer)
  * [Bert](NLP/bert-oneflow)
  * [CPT](NLP/CPT)
- Audio
  * [SincNet](Audio/SincNet)
  * [Wav2Letter](Audio/Wav2Letter)
  * [SpeakerIdentificationDemo](Audio/speaker_identification_demo)
  * [AM_MobileNet1D](Audio/AM_MobileNet1D)
- Deep Reinforcement Learning
  * [FlappyBird](DeepReinforcementLearning/FlappyBird)
- Quantization Aware Training
  * [Quantization](Quantization)

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

