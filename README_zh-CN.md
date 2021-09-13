# OneFlow-Models
**本仓库包含了基于最新OneFlow(version >= 0.5.0)实现的主流深度学习模型**

## Introduction
[English](/README.md) | **简体中文**

OneFlow-Models目录下提供了各种经典图形分类、目标检测、图像分割、对抗生成网络、自然语言处理、强化学习、量化感知学习以及语音模型的官方实现。对于每个模型，我们同时提供了模型的定义、训练以及推理的代码。并且对于每个模型，我们至少提供两个脚本`train.sh`和`infer.sh`，分别对应模型的训练和推理，便于使用者快速上手。并且保证该仓库适配OneFlow最新的API，同时提供优质的模型实现。并且与此同时我们会提供详细且高质量的学习文档，帮助使用者能够快速入手OneFlow。

## 主要特性
- 提供丰富的模型实现
- 提供对应预训练模型
- 便于上手，简单易用

## 官方模型
**快速上手**

- [demo](/demo/)

**模型库**
- 视觉任务
  - 图像分类
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
  - 视频分类
    * [TSN](Vision/classification/video/TSN)
  - 目标检测
    * [CSRNet](Vision/detection/CSRNet)
  - 语义分割
    * [FODDet](Vision/segmentation/FODDet)
    * [FaceSeg](Vision/segmentation/FaceSeg)
  - 对抗生成网络
    * [DCGAN](Vision/gan/DCGAN)
    * [SRGAN](Vision/gan/SRGAN)
    * [Pix2Pix](Vision/gan/pix2pix)
    * [CycleGAN](Vision/gan/cycleGAN)
  - 图像风格迁移
    * [FastNeuralStyle](Vision/style_transform/fast_neural_style)
  - 行人重识别
    * [bot](Vision/reid/bot)
- 自然语言处理
  * [RNN](NLP/rnn)
  * [Seq2Seq](NLP/seq2seq)
  * [LSTMText](NLP/LSTMText)
  * [TextCNN](NLP/TextCNN)
  * [Transformer](NLP/Transformer)
  * [Bert](NLP/bert-oneflow)
  * [CPT](NLP/CPT)
- 语音
  * [SincNet](Audio/SincNet)
  * [Wav2Letter](Audio/Wav2Letter)
  * [SpeakerIdentificationDemo](Audio/speaker_identification_demo)
  * [AM_MobileNet1D](Audio/AM_MobileNet1D)
- 深度强化学习
  * [FlappyBird](DeepReinforcementLearning/FlappyBird)
- 量化感知学习
  * [Quantization](Quantization)

## 安装与环境配置
**安装最新的OneFlow**

https://github.com/Oneflow-Inc/oneflow#install-with-pip-package

**环境配置**

在根目录下执行以下命令即可调用一些定制化的算子:
```bash
mkdir build
cd build
cmake ..
make -j$(nrpoc)
```
使用示例:
```bash
from ops import RoIAlign
pooler = RoIAlign(output_size=(14, 14), spatial_scale=2.0, sampling_ratio=2)
```

