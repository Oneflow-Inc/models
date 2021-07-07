# fast-neural-style :city_sunrise: :rocket:
This repository contains a oneflow implementation of an algorithm for artistic style transfer. The algorithm can be used to mix the content of an image with the style of another image. 
## Results

The mosaic, candy, udnie, and rain princess style models are fine-tuned from pytorch official models. The sketch style model is trained in oneflow.

<p align="center">
    <img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/fast_neural_style/sketch.jpeg" height="200px">
    <img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/fast_neural_style/amber.jpg" height="200px">
    <img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/fast_neural_style/amber-sketch-oneflow.jpg" height="400px">
</p>

<p align="center">
    <img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/fast_neural_style/cat.jpg" height="200px">
    <img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/fast_neural_style/cat_sketch.jpg" height="400px">
</p>

<p align="center">
    <img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/fast_neural_style/shanghai.jpeg" height="200px">
    <img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/fast_neural_style/shanghai_sketch.jpg" height="400px">
</p>

<p align="center">
    <img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/fast_neural_style/shanghai.jpeg" height="200px">
    <img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/fast_neural_style/shanghai_mosaic.jpg" height="400px">
</p>

<p align="center">
    <img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/fast_neural_style/shanghai.jpeg" height="200px">
    <img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/fast_neural_style/shanghai_udnie.jpg" height="400px">
</p>

<p align="center">
    <img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/fast_neural_style/shanghai.jpeg" height="200px">
    <img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/fast_neural_style/shanghai_candy.jpg" height="400px">
</p>

<p align="center">
    <img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/fast_neural_style/shanghai.jpeg" height="200px">
    <img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/fast_neural_style/shanghai_rain_princess.jpg" height="400px">
</p>

## Infer and Train

To get sample content, output, and style images, run
```
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/fast_neural_style/images.tar.gz
tar zxf image.tar.gz
```

For inferring, run
```
bash infer.sh
```
To customize infer process, see comment in `infer.sh`.

For training, first download coco dataset from http://msvocds.blob.core.windows.net/coco2015/test2015.zip.
Set dataset directory in train.sh. Hyperparameters can be customized. See comment in `train.sh`. Run
```
bash train.sh
```

Reference: https://github.com/pytorch/examples/tree/master/fast_neural_style