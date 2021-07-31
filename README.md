# CSRNet

## Introduction
This repo is based on: https://github.com/leeyeehoo/CSRNet-pytorch
## Installation
oneflow==0.4.0<br>
cuda==10.2
## Get started
### Dataset
ShanghaiTech Dataset:
Baidu Netdisk: 

pretrained model:

 

### Train a model
python train.py train.json val.json 0 0
### Test a model
python val.py


### Citation


    @inproceedings{li2018csrnet,
    title={CSRNet: Dilated convolutional neural networks for understanding the highly congested scenes},
    author={Li, Yuhong and Zhang, Xiaofan and Chen, Deming},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    pages={1091--1100},
    year={2018}
    }

    @inproceedings{zhang2016single,
    title={Single-image crowd counting via multi-column convolutional neural network},
    author={Zhang, Yingying and Zhou, Desen and Chen, Siqin and Gao, Shenghua and Ma, Yi},
    booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
    pages={589--597},
    year={2016}
}

#### Compare
|         | ShanghaiA MAE | ShanghaiB MAE   |
| :------ | :-----------: | ---------------:|
| PyTorch |     68.2      |      10.6       |
| OneFlow |     70.3      |       9.9       |
