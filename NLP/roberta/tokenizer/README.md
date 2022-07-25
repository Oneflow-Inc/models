# RoBERTA
## Reference

This repo is based on RobertaTokenizer from transformers, including https://github.com/huggingface/transformers/blob/master/src/transformers/models/roberta/tokenization_roberta.py, https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py, etc.

## requirement

```
pip install -r requirements.txt
```

## How to use
Example:
```
>>> from tokenizer.RobertaTokenizer import RobertaTokenizer
>>> tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
>>> tokenizer("Hello world")['input_ids']
[0, 31414, 232, 328, 2]
>>> tokenizer(" Hello world")['input_ids']
[0, 20920, 232, 2]
```
For more details about Parameters and returns, you can refer to  https://huggingface.co/transformers/model_doc/roberta.html#robertatokenizer.
