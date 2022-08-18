# ModelZoo Model List

| Model | Parameters | Note |
| --- | --- | --- |
| **BERT** |  |  |
| bert-small-uncased | L=6,H=768,A=12 |  |
| bert-base-uncased | L=12,H=768,A=12 |  |
| bert-large-uncased | L=24,H=1024,A=16 |  |
| alibaba-pai/pai-bert-base-zh | L=6,H=768,A=12 |  Pretrain BERT w/ Chinese datasets|
| alibaba-pai/pai-bert-small-zh | L=4,H=312,A=12 |  |
| alibaba-pai/pai-bert-tiny-zh | L=2,H=128,A=2 |  |
| **Mengzi （Langboat)** |  |  |
| langboat/mengzi-bert-base | L=12,H=768,A=12| Pretrain BERT w/ Chinese datasets|
| langboat/mengzi-bert-base-fin | L=12,H=768,A=12 | Pretrain BERT w/ Finance datasets
| **DKPLM（知识预训练）** |  |  |
| alibaba-pai/pai-dkplm-medical-small-zh | L=4,H=768,A=12 | Pretrain BERT w/ Medical KG|
| alibaba-pai/pai-dkplm-medical-base-zh | L=12,H=768,A=12 |  |
| alibaba-pai/pai-dkplm-medical-large-zh | 待发布 |  |
| alibaba-pai/pai-dkplm-small-zh | 待发布 | Pretrain BERT w/ General KG|
| alibaba-pai/pai-dkplm-base-zh | 待发布 |  |
| alibaba-pai/pai-dkplm-large-zh | 待发布 |  |
| alibaba-pai/pai-dkplm-1.3b-zh | 待发布 |  |
| alibaba-pai/pai-dkplm-13b-zh | 待发布 |  |
| **GEEP（加速版BERT）** |  |  |
| alibaba-pai/geep-bert-base-zh |  |  |
| alibaba-pai/geep-bert-large-zh |  |  |
| **RoBERTa** |  |  |
| hfl/chinese-roberta-wwm-ext | L=12,H=768,A=12 |  |
| hfl/chinese-roberta-wwm-ext-large | L=24,H=1024,A=16 |  |
| roberta-base-en | L=12,H=768,A=12 |  |
| roberta-large-en | L=24,H=1024,A=16 |  |
| **MacBERT** |  |  |
| hfl/macbert-base-zh | L=12,H=768,A=12 |  |
| hfl/macbert-large-zh | L=24,H=1024,A=16 |  |
| **Generation** |  |  |
| alibaba-pai/gpt2-chitchat-zh | L=10,H=768,A=12 | for Chinese dialogue |
| alibaba-pai/pegasus-summary-generation-en | L=16,H=1024,A=16 | for English text summarization |
| alibaba-pai/mt5-title-generation-zh | L=12,H=1024,A=12 | for Chinese News title generation |

# cli使用方式
```bash
$ easynlp \
   --mode=train \
   --worker_gpu=1 \
   --tables=train.tsv,dev.tsv \
   --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
   --first_sequence=sent1 \
   --label_name=label \
   --label_enumerate_values=0,1 \
   --checkpoint_dir=./classification_model \
   --epoch_num=1  \
   --sequence_length=128 \
   --app_name=text_classify \
   --user_defined_parameters='pretrain_model_name_or_path=bert-small-uncased'
```

# 代码使用方式

```python
args = initialize_easynlp()
train_dataset = ClassificationDataset(xxx)
model = SequenceClassification(pretrained_model_name_or_path='bert-small-uncased')
Trainer(model=model,  train_dataset=train_dataset).train()
```



