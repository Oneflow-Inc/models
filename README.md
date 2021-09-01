# models
Models and examples implement with OneFlow(version >= 0.5.0).

## List of Models

### Image Classification
* [Lenet](https://github.com/Oneflow-Inc/models/tree/main/quick_start_demo_lenet)
* [Alexnet](https://github.com/Oneflow-Inc/models/tree/main/alexnet)
* [VGG16/19](https://github.com/Oneflow-Inc/models/tree/main/vgg)
* [Resnet50](https://github.com/Oneflow-Inc/models/tree/main/resnet50)
* [InceptionV3](https://github.com/Oneflow-Inc/models/tree/main/inception_v3)
* [Densenet](https://github.com/Oneflow-Inc/models/tree/main/densenet)
* [Resnext50_32x4d](https://github.com/Oneflow-Inc/models/tree/main/resnext50_32x4d)
* [Shufflenetv2](https://github.com/Oneflow-Inc/models/tree/main/shufflenetv2)
* [MobilenetV2](https://github.com/Oneflow-Inc/models/tree/main/mobilenetv2)
* [mobilenetv3](https://github.com/Oneflow-Inc/models/tree/main/mobilenetv3)
* [Ghostnet](https://github.com/Oneflow-Inc/models/tree/main/ghostnet)
* [Repvgg](https://github.com/Oneflow-Inc/models/tree/main/repvgg)
* [DLA](https://github.com/Oneflow-Inc/models/tree/main/DLA)
* [PoseNet](https://github.com/Oneflow-Inc/models/tree/main/poseNet)
* [Scnet](https://github.com/Oneflow-Inc/models/tree/main/scnet)

### Video Classification
* [TSN](https://github.com/Oneflow-Inc/models/tree/main/TSN)

### Object Detection
* [CSRNet](https://github.com/Oneflow-Inc/models/tree/main/CSRNet)

### Semantic Segmentation
* [FODDet](https://github.com/Oneflow-Inc/models/tree/main/FODDet)
* [FaceSeg](https://github.com/Oneflow-Inc/models/tree/main/FaceSeg)

### Generative Adversarial Networks
* [DCGAN](https://github.com/Oneflow-Inc/models/tree/main/DCGAN)
* [SRGAN](https://github.com/Oneflow-Inc/models/tree/main/SRGAN)
* [Pix2Pix](https://github.com/Oneflow-Inc/models/tree/main/pix2pix)
* [CycleGAN](https://github.com/Oneflow-Inc/models/tree/main/cycleGAN)

### Neural Style
* [FastNeuralStyle](https://github.com/Oneflow-Inc/models/tree/main/fast_neural_style)

### Deep Reinforcement Learning
* [FlappyBird](https://github.com/Oneflow-Inc/models/tree/main/FlappyBird)

### Person Re-identification
* [Reid](https://github.com/Oneflow-Inc/models/tree/main/reid)

### Natural Language Processing
* [RNN](https://github.com/Oneflow-Inc/models/tree/main/rnn)
* [Seq2Seq](https://github.com/Oneflow-Inc/models/tree/main/seq2seq)
* [LSTMText](https://github.com/Oneflow-Inc/models/tree/main/LSTMText)
* [TextCNN](https://github.com/Oneflow-Inc/models/tree/main/TextCNN)
* [Transformer](https://github.com/Oneflow-Inc/models/tree/main/Transformer)
* [Bert](https://github.com/Oneflow-Inc/models/tree/main/bert-oneflow)

### Audio
* [SincNet](https://github.com/Oneflow-Inc/models/tree/main/SincNet)
* [Wav2Letter](https://github.com/Oneflow-Inc/models/tree/main/Wav2Letter)
* [SpeakerIdentificationDemo](https://github.com/Oneflow-Inc/models/tree/main/speaker_identification_demo)

### Quantization Aware Training
* [Quantization](https://github.com/Oneflow-Inc/models/tree/main/quantization)

## Quick Start

### Install Oneflow 
https://github.com/Oneflow-Inc/oneflow#install-with-pip-package

### Build custom ops from source
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
