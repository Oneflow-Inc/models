# Face segmentation

This repo is about face segmentation based on LinkNet34, our work is inspired by this PyTorch [implementaion](https://github.com/JiaojiaoYe1994/face-segmentation).


### Prerequisites

```
pip3 install -r requirements.txt
```


### Download Pretrain Models

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/seg/linknet_oneflow_model.zip
```


## Inference to segment face for webcam

```bash
bash infer.sh
```


## Train 

### Data preparation

We combine several dataset and preprocess them, for training and evaluating our 
face segmentation model. We provide the final compressed dataset, please download dataset from [here](https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/faceseg_data.zip) , unzip it and put it in `./data`

### Run Oneflow Training script

coming soon!
