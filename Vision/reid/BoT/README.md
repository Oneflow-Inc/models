# Re-ID

## Introduction
This repo provides one method for Person Re-identification task ([code](https://github.com/Oneflow-Inc/oneflow_vision_model/tree/main/Re-ID) in lazy mode).
## Get started


### Prerequisites

```
pip3 install -r requirements.txt
```



### Data preparation

We use [Market1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf) to train and evaluate our Re-ID models.
Please create a folder named `datasets` and then download it [here](https://oneflow-static.oss-cn-beijing.aliyuncs.com/Dataset/market1501.zip) and unzip in `datasets` first.



### Train a model

To train a model, run ```sh train.sh```


Our model is trained using [ResNet50](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/reid/resnet50_pretrained_model.zip) as backbones


### Evaluate a model
To evaluate a model, run ```sh infer.sh```
