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
欢迎体验OneFlow的入门[demo](/demo/)

## 相关文档
可以通过我们的[官方文档](https://oneflow-models.readthedocs.io/en/latest/)了解OneFlow在不同领域的模型实现

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

