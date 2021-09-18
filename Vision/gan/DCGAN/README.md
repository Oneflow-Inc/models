# DCGAN

DCGAN is an unconditional image generation method ([code](https://github.com/Oneflow-Inc/oneflow_vision_model/tree/main/DCGAN) in lazy mode), our code is inspired by [TensorFlow Tutorial](https://tensorflow.google.cn/tutorials/generative/dcgan).

Please make sure your working directory is `models/DCGAN/`.

## Train on [mnist](http://yann.lecun.com/exdb/mnist/) Dataset
### Run Oneflow Training script
Eager Mode
```bash
bash eager/train.sh
```

Graph Mode
```bash
bash graph/train.sh
```

## Inference
### Run Oneflow Inference script

```bash
bash test/test_of_dcgan.sh
```

## Check
### Run Oneflow Check script

```bash
bash check/check.sh
```

### Inference results

#### Eager Mode

![test_images](https://i.loli.net/2021/08/11/tgLG975APOTFual.png)

#### Graph Mode

![image_100](https://i.loli.net/2021/08/11/LZ8BRuTEcNxgHjX.png)