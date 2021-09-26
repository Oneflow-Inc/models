# OneFlow-Models
**本仓库包含了基于最新OneFlow(version >= 0.5.0)实现的主流深度学习模型**

## Introduction
[English](/README.md) | **简体中文**

OneFlow-Models目录下提供了各种经典图形分类、目标检测、图像分割、对抗生成网络、自然语言处理、强化学习、量化感知学习以及语音模型的官方实现。对于每个模型，我们同时提供了模型的定义、训练以及推理的代码。并且对于每个模型，我们至少提供两个脚本`train.sh`和`infer.sh`，分别对应模型的训练和推理，便于使用者快速上手。并且保证该仓库适配OneFlow最新的API，同时提供优质的模型实现。并且与此同时我们会提供详细且高质量的学习文档，帮助使用者能够快速入手OneFlow。

## 主要特性
- 提供丰富的模型实现
- 提供对应预训练模型
- 便于上手，简单易用

## 快速上手
欢迎体验OneFlow的入门Demo
- **图像分类:** [LeNet](Demo/quick_start_demo_lenet/lenet.py)
- **说话人识别:** [Speaker Identification](Demo/speaker_identification_demo)

## 模型目录
<details>
<summary> <b> 图像分类 </b> </summary>

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
  - [ViT](https://github.com/Oneflow-Inc/models/tree/main/Vision/classification/image/ViT)

</details>

<details>
<summary> <b> 视频分类 </b> </summary>

- [TSN](https://github.com/Oneflow-Inc/models/tree/main/Vision/classification/video/TSN)

</details>


<details>
<summary> <b> 目标检测 </b> </summary>
  
- [CSRNet](https://github.com/Oneflow-Inc/models/tree/main/Vision/detection/CSRNet)

</details>

<details>
<summary> <b> 语义分割 </b> </summary>

- [FODDet](https://github.com/Oneflow-Inc/models/tree/main/Vision/segmentation/FODDet)
- [FaceSeg](https://github.com/Oneflow-Inc/models/tree/main/Vision/segmentation/FaceSeg)

</details>

<details>
<summary> <b> 对抗生成网络 </b> </summary>

- [DCGAN](https://github.com/Oneflow-Inc/models/tree/main/Vision/gan/DCGAN)
- [SRGAN](https://github.com/Oneflow-Inc/models/tree/main/Vision/gan/SRGAN)
- [Pix2Pix](https://github.com/Oneflow-Inc/models/tree/main/Vision/gan/Pix2Pix)
- [CycleGAN](https://github.com/Oneflow-Inc/models/tree/main/Vision/gan/CycleGAN)

</details>

<details>
<summary> <b> 图像风格迁移 </b> </summary>

- [FastNeuralStyle](https://github.com/Oneflow-Inc/models/tree/main/Vision/style_transform/fast_neural_style)

</details>


<details>
<summary> <b> 行人重识别 </b> </summary>

- [BoT](https://github.com/Oneflow-Inc/models/tree/main/Vision/reid/BoT)

</details>


<details>
<summary> <b> 自然语言处理 </b> </summary>

- [RNN](https://github.com/Oneflow-Inc/models/tree/main/NLP/rnn)
- [Seq2Seq](https://github.com/Oneflow-Inc/models/tree/main/NLP/seq2seq)
- [LSTMText](https://github.com/Oneflow-Inc/models/tree/main/NLP/LSTMText)
- [TextCNN](https://github.com/Oneflow-Inc/models/tree/main/NLP/TextCNN)
- [Transformer](https://github.com/Oneflow-Inc/models/tree/main/NLP/Transformer)
- [Bert](https://github.com/Oneflow-Inc/models/tree/main/NLP/bert-oneflow)
- [CPT](https://github.com/Oneflow-Inc/models/tree/main/NLP/CPT)
</details>

<details>
<summary> <b> 语音 </b> </summary>

- [SincNet](https://github.com/Oneflow-Inc/models/tree/main/Audio/SincNet)
- [Wav2Letter](https://github.com/Oneflow-Inc/models/tree/main/Audio/Wav2Letter)
- [AM_MobileNet1D](https://github.com/Oneflow-Inc/models/tree/main/Audio/AM-MobileNet1D)
- [Speech-Emotion-Analyer](https://github.com/Oneflow-Inc/models/tree/main/Audio/Speech-Emotion-Analyzer)
</details>

<details>
<summary> <b> 深度强化学习 </b> </summary>

- [FlappyBird](https://github.com/Oneflow-Inc/models/tree/main/DeepReinforcementLearning/FlappyBird)
</details>

<details>
<summary> <b> 量化感知学习 </b> </summary>

- [Quantization](https://github.com/Oneflow-Inc/models/tree/main/Quantization)
</details>

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

