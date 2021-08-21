# SOD based on LDFNet

This work is about Salient Object Detection based on LDFNet, this code is inspired by this PyTorch version [implementaion](https://github.com/weijun88/LDF).

## Prerequisites

```
pip3 install -r requirements.txt
```

## Data preparation
Download the dataset and unzip them into according data folder.

-[DUTS-TR](http://saliencydetection.net/duts/download/DUTS-TR.zip) / 
-[DUTS-TE](http://saliencydetection.net/duts/download/DUTS-TE.zip)


## Download trained Models

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/seg/LDFnet/LDF_model.zip
```


## Train and Test
- If you want to train the model by yourself, please download the [pretrained model](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/seg/LDFnet/resnet50-19c8e357.zip) into `./model/pretrained` folder
- Split the ground truth into body map and detail map, which will be saved into `data/DUTS-TR/body-origin` and `data/DUTS-TR/detail-origin`
```shell
    python3 utils.py
```
- Train the model and get the predicted maps, which will be saved into `eval/DUTS-TR`
```shell
    cd model
    python3 train.py
    python3 test.py
```

 
### Demonstration
The following displays a sample about SOD.

<img src="https://github.com/ClimBin/LDFnet/blob/main/maps/sample.jpg"/>
<img src="https://github.com/ClimBin/LDFnet/blob/main/maps/sample.png"/>
