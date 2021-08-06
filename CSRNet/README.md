# CSRNet

## Introduction
This repo is based on: https://github.com/leeyeehoo/CSRNet-pytorch


## Prerequisites

```
pip3 install -r requirements.txt
```

## Get started
### Dataset
ShanghaiTech crowd counting dataset contains 1198 an-notated images with a total amount of 330,165 persons.
This dataset consists of two parts as Part A containing 482 images with highly congested scenes randomly downloaded
from the Internet while Part B includes 716 images with relatively sparse crowd scenes taken from streets in Shang-
hai.[ShanghaiTech ](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/CSRNet/Shanghai_dataset.rar) Dataset are provided.


### Download Pretrain Models

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/CSRNet/Shanghai_BestModelA.rar
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/CSRNet/Shanghai_BestModelB.rar
```

## Follow the val.sh to try the validation
First, modify the address of the Pretrain Models to be used in the val.py file
```bigquery
checkpoint = flow.load('checkpoint/Shanghai_BestModelA/shanghaiA_bestmodel')
```
```bash
bash infer.sh
```

### Train a model
```
python train.py part_A_train.json part_A_val.json 0 0
```

```bash
bash train.sh
```
### predicted values on a single image:
```
python test.py
```

```bash
bash infer.sh
```

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
|         | ShanghaiA MAE |  ShanghaiB MAE   |
| :------ | :-----------: | ---------------:|
| PyTorch |     68.2      |      10.6       |
| OneFlow |     70.3      |       9.9       |
