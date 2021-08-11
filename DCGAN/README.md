# DCGAN

DCGAN is an unconditional image generation method ([code](https://github.com/Oneflow-Inc/oneflow_vision_model/tree/main/DCGAN) in lazy mode), our code is inspired by [TensorFlow Tutorial](https://tensorflow.google.cn/tutorials/generative/dcgan).

Please make sure your working directory is `models/DCGAN/`.

## Train on [mnist](http://yann.lecun.com/exdb/mnist/) Dataset
### Run Oneflow Training script

```bash
bash eager/train_of_dcgan.sh
```

## Inference
### Run Oneflow Inference script

```bash
bash test/test_of_dcgan.sh
```

### Run Oneflow Training script: Graph

```bash
bash graph/train_of_dcgan_graph.sh
```

### Run Oneflow Check script

```bash
bash check/check.sh
```
Inference results

![](test_images.png)