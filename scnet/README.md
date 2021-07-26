# Scnet

Scnet is a vehicle color classification network based on scloss, which can be easily combined with different backbone networks.

## Train on [hustcolor](http://cloud.eic.hust.edu.cn:8071/~pchen/color.rar) Dataset

### Prepare Traning Data
The Vehicle Color recognition Dataset contains 15601 vehicle images in eight colors, which are black, blue, cyan, gray, green, red, white and yellow. The images are taken in the frontal view captured by a high-definition camera with the resolution of 1920×1080 on the urban road. The collected data set is very challenging due to the noise caused by illumination variation, haze, and over exposure.Datasets in the OfRecord format are provided.

### Run Oneflow Training script
Visdom is installed to visualize the training model, and you can enter http://localhost:8097/ get the training curve.

```
pip3 install -r requirements.txt --user
python -m visdom.server
```
```bash
bash train_oneflow.sh
```
## Inference on Single Image
### Download pretrained model
The pretrained [model](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/transfer_learning/scnet_acc_0.947254.zip) on hustcolor.

```bash
bash infer.sh
```
 ![image](https://github.com/XinYangDong/models/blob/main/scnet/data/red_prediction.png)
### Accracy of model
|         | val(Top1) |
| :-----: | :-----------------: |
| resnet  |        0.925        |
| scnet   |        0.947        |
