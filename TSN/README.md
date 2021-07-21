# TSN

This repo provides one method for Person Action Recognition task.

## Get started
Prerequisites
pip3 install -r requirements.txt

## Data preparation
We use Kinetics400 to train and evaluate our models. Please refer to PREPARING_KINETICS400.md for data prepration.


##Train a model

To train a model, run sh train.sh

Our model is trained using ResNet50 as backbones
##Evaluate a model

To evaluate a model, run sh infer.sh

## accuracy
            oneflow     torch
top1 acc    0.6373      0.6373
top5 acc    0.6785      0.6785

