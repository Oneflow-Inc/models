# Vision Transformer

## Train on [imagenette](https://github.com/fastai/imagenette) Dataset

### Prepare Traning Data

#### Download Ofrecord

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenette_ofrecord.tar.gz
tar zxf imagenette_ofrecord.tar.gz
```

### Run Oneflow Training script

```bash
bash train.sh
```


## Inference on Single Image

```bash
bash infer.sh
```

## Pretrained Model Weight
### Built Model
```
vit_b_16_224
vit_b_16_384
vit_b_32_224
vit_b_32_384
vit_l_16_384
vit_l_32_384
```
**Note that you can pass `--model_arch=str` in your training scripts or train.sh for training different model**

### Pretrained Model List
| Model | Patch_size | Image Size | Dataset | Acc1 | Pretrained Weight |
|:---:|:---:|:---:|:---:|:---:|:---:|
|vit_b_16_384| 16 | 384 | imagenet2012 | 83.90 | [checkpoint]() |
|vit_b_32_384| 16 | 384 | imagenet2012 | 81.14 | [checkpoint]() |
|vit_l_16_384| 16 | 384 | imagenet2012 | 84.94 | [checkpoint]() |
|vit_l_32_384| 16 | 384 | imagenet2012 | 81.03 | [checkpoint]() |

**Note that you can download any checkpoint above and modify the infer.sh or train.sh for training or testing**


## Acknowledgement
- [Vision Transformer](https://github.com/asyml/vision-transformer-pytorch)