## UNIT: UNsupervised Image-to-image Translation Networks

### Paper

[Ming-Yu Liu, Thomas Breuel, Jan Kautz, "Unsupervised Image-to-Image Translation Networks" NIPS 2017 Spotlight, arXiv:1703.00848 2017](https://arxiv.org/abs/1703.00848)

Modified from https://github.com/mingyuliutw/UNIT

### License
怎么写


### Requirments
 
- Python package
  - `pip install flowvision==0.0.5`
  - `pip install pyyaml`
  - `pip install tensorboard tensorboardX`

### Testing

First, download converted pretrained models for the gta2cityscape task and put them in `models` folder.

#### Pretrained models 

|  Dataset    | Model Link     |
|-------------|----------------|
| gta2cityscape |   [model](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/gan/unit_gta2city.weight.zip) | 

| Vgg16     |
|----------------|
|   [model](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/gan/vgg16_fast_neural_style.zip) | 
