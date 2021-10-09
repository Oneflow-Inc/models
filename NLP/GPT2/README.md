# GPT-2

This repo privides code for the paper "[Language Models are Unsupervised Multitask Learners](https://blog.openai.com/better-language-models/)", which is based on https://github.com/huggingface/transformers.

## Train from scratch
train.sh script will help you train GPT2 from scratch. You can train on your own corpus and change hyperparameters in train.sh file.
```bash
bash train.sh
```

## Finetune
You can finetune pre-trained model on your own corpus via fintune.sh file. Please modify hyperparameters in the finetune.sh file.

```bash
bash finetune.sh
```

## Convert PyTorch checkpoint to OneFlow checkpoint
```python
from model_config import GPT2Config
from model import GPT2LMHeadModel
from convert_pt_ckpt_to_of import convert_pt_checkpoint_to_of

config = GPT2Config()
model = GPT2LMHeadModel(config)

convert_pt_checkpoint_to_of(model, "gpt2_model.pt", "gpt2_oneflow_model")
```

## Inference
infer.sh script privides a simple interface to invoke well-trained GPT2 models.
```bash
bash infer.sh
```