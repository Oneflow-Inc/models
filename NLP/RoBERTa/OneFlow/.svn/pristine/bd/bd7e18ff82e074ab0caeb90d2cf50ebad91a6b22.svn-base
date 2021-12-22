# RoBERTa权重迁移

- 权重迁移由`move_weight.py`完成，测试方法为对同样的随机输入数据计算L1损失。
    - `base_weight_utils.py`包含了模型基础构成部分的迁移函数
    - `roberta_wegiht_utils.py`包含Roberta及其衍生模型相关的迁移函数
    - `roberta.py`包含了用于迁移权重的类。每种衍生模型的迁移都继承自一个基础类，方便针对不同类进行权重迁移。
        - 暂时只编写了预训练权重需要的`RobertaForMaskedLM`和`RobertaForSequenceClassification`两种模型的迁移类。
- 已经在仅有`RobertaModel`的情况下对transformers提供的六种预训练模型完成了权重转换，保存在`robertaonly_pretrain_oneflow/`下。
    - 衍生类的迁移待测试。
        - `RobertaForMaskedLM`的迁移会有略大一点的误差（1e-4级），原因未知

## 库

- json

## 预训练模型：

相关信息可以参照[transformers文档](https://huggingface.co/transformers/pretrained_models.html)和[huggingface官网](https://huggingface.co/models)
- roberta-base (`RobertaForMaskedLM`)
- roberta-large (`RobertaForMaskedLM`)
- roberta-large-mnli (`RobertaForSequenceClassification`)
- distilroberta-base (`RobertaForMaskedLM`)
- roberta-base-openai-detector (`RobertaForSequenceClassification`)
- roberta-large-openai-detector (`RobertaForSequenceClassification`)

## 保存格式

- 以预训练模型`roberta-base`为例，默认情况下，预训练模型保存在`/remote-home/share/shxing/roberta_pretrain_oneflow/roberta-base/weights`下，并且将参数保存在`/remote-home/share/shxing/roberta_pretrain_oneflow/roberta-base/parameters.json`中。
    - 对于单独的`Roberta`模型，我们单独保存了相应的参数以供微调，保存在`robertaonly_pretrain_oneflow`下
- 在加载预训练模型时，可参考以下方式读取权重和参数：

```python
import json
import oneflow as flow

pretrain_dir = "/remote-home/share/shxing/robertaonly_pretrain_oneflow/roberta-base/weights"
kwargs_path = "/remote-home/share/shxing/robertaonly_pretrain_oneflow/roberta-base/parameters.json"

with open(kwargs_path, "r") as f:
    kwargs = json.load(f)
model = Roberta(**kwargs)
model.load_state_dict(flow.load(pretrain_dir))
```