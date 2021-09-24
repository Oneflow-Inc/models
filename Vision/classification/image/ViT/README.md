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
### Model List
| Model | Patch_size | Image Size | Dataset | Acc1 | Pretrained Weight |
|:---:|:---:|:---:|:---:|:---:|:---:|
|vit_b_16_224| 16 | 224 | imagenet2012 | 83.90 | - |
|vit_b_16_384| 16 | 384 | imagenet2012 | 83.90 | [checkpoint]() |
|vit_b_32_224| 16 | 224 | imagenet2012 | 83.90 | - |
|vit_b_32_384| 16 | 384 | imagenet2012 | 83.90 | [checkpoint]() |
|vit_l_16_384| 16 | 384 | imagenet2012 | 83.90 | [checkpoint]() |
|vit_l_32_384| 16 | 384 | imagenet2012 | 83.90 | [checkpoint]() |

**Note that you can download any checkpoint above and modify the infer.sh or train.sh for training or testing**


## Acknowledgement
- [Vision Transformer](https://github.com/asyml/vision-transformer-pytorch)