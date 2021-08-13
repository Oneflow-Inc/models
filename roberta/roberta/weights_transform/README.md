# RoBERTa权重迁移

权重迁移由`move_weight.py`和`roberta_weight_utils.py`完成，测试方法为对同样的随机输入数据计算L1损失。
已经在仅有`RobertaModel`的情况下对transformers提供的六种预训练模型完成了权重转换。

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

以预训练模型`roberta-base`为例，默认情况下，预训练模型保存在`roberta_pretrain_oneflow/roberta-base`下，并且将参数保存在`roberta_pretrain_oneflow/roberta-base/roberta-base.json`中。在加载预训练模型时，可参考以下方式读取权重和参数：

```python
import json
import oneflow as flow

pretrain_dir = "roberta_pretrain_oneflow/roberta-base"
kwargs_path = "roberta_pretrain_oneflow/roberta-base/roberta-base.json"

with open(kwargs_path, "r") as f:
    kwargs = json.load(f)
model = Roberta(**kwargs)
model.load_state_dict(flow.load(pretrain_dir))
```