## Vision Transformer (ViT)
Load Pretrained ViT model and fine-tune on ImageNet2012 Dataset using **OneFlow**

## Contents
- [x] Load Pretrained Weight and run val on `ImageNet_val_2012`
- [ ] Fine-Tune on `ImageNet_train_2012`

## Usage
### 1. Prepare Training Data And Pretrained Models
- Prepare ImageNet 2012 Dataset as follows:
```bash
$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```

- Downlaod Pretrained Weight
```bash
$ wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/ViT-OneFlow/ViT-B_16_oneflow.zip
$ unzip ViT-B_16_oneflow.zip
```

### 3. Run OneFlow Validation Scripts
```bash
$ bash eval.sh
```

## Model Zoo
| upstream    | model    | dataset      | orig. jax acc  |  oneflow acc  | model link                                                                                                                                                   |
|:------------|:---------|:-------------|---------------:|--------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------|
| imagenet21k | ViT-B_16 | imagenet2012 |     84.62      |     83.90     | [checkpoint](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/ViT-OneFlow/ViT-B_16_oneflow.zip) |
