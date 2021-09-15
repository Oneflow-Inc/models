# OneFlow-Models
**Models and examples implement with OneFlow(version >= 0.5.0).**

## Introduction
**English** | [简体中文](/README_zh-CN.md)

OneFlow-Models is an open source repo which contains official implementation of different models built on OneFlow. In each model, we provide at least two scripts `train.sh` and `infer.sh` for a quick start. For each model, we provide a detailed `README` to introduce the usage of this model.

## Features
- various models and pretrained weight
- easy use for beginners

## Quick Start
Please check our the following **demos** for a quick start
- **image classification** [quick start lenet demo](Demo/quick_start_demo_lenet/lenet.py)
- **speaker recognition** [speaker identification demo](Demo/speaker_identification_demo)

## Model List
<details>
<summary> <b> Computer Vision </b> </summary>

- **Image Classification**
  - [Lenet](https://github.com/Oneflow-Inc/models/blob/main/Demo/quick_start_demo_lenet/lenet.py)
  - [Alexnet](https://github.com/Oneflow-Inc/models/tree/main/Vision/classification/image/alexnet)
  - [VGG16/19](https://github.com/Oneflow-Inc/models/tree/main/Vision/classification/image/vgg)
  - [Resnet50](https://github.com/Oneflow-Inc/models/tree/main/Vision/classification/image/resnet50)
  - [InceptionV3](https://github.com/Oneflow-Inc/models/tree/main/Vision/classification/image/inception_v3)
  - [Densenet](https://github.com/Oneflow-Inc/models/tree/main/Vision/classification/image/densenet)
  - [Resnext50_32x4d](https://github.com/Oneflow-Inc/models/tree/main/Vision/classification/image/resnext50_32x4d)
  - [Shufflenetv2](https://github.com/Oneflow-Inc/models/tree/main/Vision/classification/image/shufflenetv2)
  - [MobilenetV2](https://github.com/Oneflow-Inc/models/tree/main/Vision/classification/image/mobilenetv2)
  - [mobilenetv3](https://github.com/Oneflow-Inc/models/tree/main/Vision/classification/image/mobilenetv3)
  - [Ghostnet](https://github.com/Oneflow-Inc/models/tree/main/Vision/classification/image/ghostnet)
  - [RepVGG](https://github.com/Oneflow-Inc/models/tree/main/Vision/classification/image/repvgg)
  - [DLA](https://github.com/Oneflow-Inc/models/tree/main/Vision/classification/image/DLA)
  - [PoseNet](https://github.com/Oneflow-Inc/models/tree/main/Vision/classification/image/poseNet)
  - [Scnet](https://github.com/Oneflow-Inc/models/tree/main/Vision/classification/image/scnet)
  - [Mnasnet](https://github.com/Oneflow-Inc/models/tree/main/Vision/classification/image/mnasnet)

- **Video Classification**
  - [TSN](https://github.com/Oneflow-Inc/models/tree/main/Vision/classification/video/TSN)

- **Object Detection**
  - [CSRNet](https://github.com/Oneflow-Inc/models/tree/main/Vision/detection/CSRNet)

- **Semantic Segmentation**
  - [FODDet](https://github.com/Oneflow-Inc/models/tree/main/Vision/segmentation/FODDet)
  - [FaceSeg](https://github.com/Oneflow-Inc/models/tree/main/Vision/segmentation/FaceSeg)

- **Generative Adversarial Networks**
  - [DCGAN](https://github.com/Oneflow-Inc/models/tree/main/Vision/gan/DCGAN)
  - [SRGAN](https://github.com/Oneflow-Inc/models/tree/main/Vision/gan/SRGAN)
  - [Pix2Pix](https://github.com/Oneflow-Inc/models/tree/main/Vision/gan/Pix2Pix)
  - [CycleGAN](https://github.com/Oneflow-Inc/models/tree/main/Vision/gan/CycleGAN)

- **Neural Style Transform**
  - [FastNeuralStyle](https://github.com/Oneflow-Inc/models/tree/main/Vision/style_transform/fast_neural_style)

- **Person Re-identification**
  - [BoT](https://github.com/Oneflow-Inc/models/tree/main/Vision/reid/BoT)

  </details>


<details>
<summary> <b> Natural Language Processing </b> </summary>

- [RNN](https://github.com/Oneflow-Inc/models/tree/main/NLP/rnn)
- [Seq2Seq](https://github.com/Oneflow-Inc/models/tree/main/NLP/seq2seq)
- [LSTMText](https://github.com/Oneflow-Inc/models/tree/main/NLP/LSTMText)
- [TextCNN](https://github.com/Oneflow-Inc/models/tree/main/NLP/TextCNN)
- [Transformer](https://github.com/Oneflow-Inc/models/tree/main/NLP/Transformer)
- [Bert](https://github.com/Oneflow-Inc/models/tree/main/NLP/bert-oneflow)
- [CPT](https://github.com/Oneflow-Inc/models/tree/main/NLP/CPT)

</details>

<details>
<summary> <b> Audio </b> </summary>

- [SincNet](https://github.com/Oneflow-Inc/models/tree/main/Audio/SincNet)
- [Wav2Letter](https://github.com/Oneflow-Inc/models/tree/main/Audio/Wav2Letter)
- [AM_MobileNet1D](https://github.com/Oneflow-Inc/models/tree/main/Audio/AM-MobileNet1D)

</details>

<details>
<summary> <b> Deep Reinforcement Learning </b> </summary>

- [FlappyBird](https://github.com/Oneflow-Inc/models/tree/main/DeepReinforcementLearning/FlappyBird)

</details>

<details>
<summary> <b> Quantization Aware Training </b> </summary>

- [Quantization](https://github.com/Oneflow-Inc/models/tree/main/Quantization)

</details>

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

