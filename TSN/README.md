# TSN

This repo provides one method for Person Action Recognition task.

## Get started
Prerequisites
```bash
pip3 install -r requirements.txt
```

## Data preparation
We use Kinetics400 to train and evaluate our models. Please download it here and unzip in current path first.
```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/TSN/kinetics400.zip
unzip kinetics400.zip
```

##Train a model

To train a model, run sh train.sh

Our model is trained using ResNet50 as backbones
##Evaluate a model

To evaluate a model, run sh infer.sh
