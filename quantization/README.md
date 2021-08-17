## 基于OneFlow实现量化感知训练

### 代码结构

```markdown
- quantization_ops 伪量化OP实现
    - q_module.py 实现了Qparam类来管理伪量化参数和操作和QModule基类管理伪量化OP的实现
    - conv.py 继承QModule基类，实现卷积的伪量化实现
    - linear.py 继承QModule基类，实现全连接层的伪量化实现
    - ...
- models 量化模型实现
    - q_alexnet.py 量化版AlexNet模型
- quantization_aware_training.py 量化训练实现
- quantization_infer.py 量化预测实现
- train.sh 量化训练脚本
- infer.sh 量化预测脚本
```

### 实验

- 比特数为~代表全精度浮点训练
- 不是PerLayer的量化就是PerChannel的量化
- 数据集是ImageNet的一个子集
- 20个epoch可能模型还没有完全收敛，但能大致看出来量化感知训练在8Bit时会不会严重掉点

|模型|Epoch|量化比特数|量化规则|量化方法|是否是PerLayer|精度|
|--|--|--|--|--|--|--|
|AlexNet|20|~|xx|xx|xx|0.721939|
|AlexNet|20|8|google|symmetric|Yes|0.717857|
|AlexNet|20|8|google|symmetric|No|0.732143|
|AlexNet|20|8|google|affine|Yes|0.737245|
|AlexNet|20|8|google|affine|No|0.713265|
|AlexNet|20|4|google|symmetric|Yes|0.579082|
|AlexNet|20|4|google|symmetric|No|0.577551|
|AlexNet|20|4|google|affine|Yes|0.583673|
|AlexNet|20|4|google|affine|No|0.634949|
|AlexNet|20|8|cambricon|symmetric|Yes|0.730867|
|AlexNet|20|4|cambricon|symmetric|Yes|0.491582|


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
