# CPT

Pre-Training for Chinese Language Understanding and Generation. This model is based on [`BartModel`](https://github.com/huggingface/transformers/blob/master/src/transformers/models/bart/modeling_bart.py) of `transformers`. 

## Model

CPT supports both understanding and generation task. CPT has two kinds of decoders: decoder for understanding (DU) and decoder for generating (DG). When we only use DU, the model is similar to BERT while use DG only, it is similar to BART. CPT model is placed in `models/`, and you can set different `cls_mode` to run different tasks:

- `cls_mode = 1`: Only encoder, or DU only.
- `cls_mode = 2`: Only decoder, or DG only.
- `cls_mode = 3`: Both encoder and decoder.

We implemented some extra functions here:

<!-- - `LayerNorm` in `models/dev_ops.py` -->
- `tensor_unique` in `models/bart_utils.py`
- `position_scores` in `models/bert_utils.py`

## Train

We use CLUE(Chinese GLUE)-AFQMC to test our CPT model. You can check [this site](https://www.cluebenchmarks.com/introduce.html) and [this site](https://github.com/CLUEbenchmark/CLUE) for details.

After training, CPT can figure out if the two input sentences are of the same meaning.

First, get the dataset:
```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/nlp/CPT/clue_afqmc.tar.gz
tar -xzf clue_afqmc.tar.gz
```

The dataset contains two json files `train.json` and `eval.json`.

Then download the pretrained model `cpt-base`, which contains 12 encoder layers and 2 decoder layers:

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/nlp/CPT/cpt-base.tar.gz
tar -xzf cpt-base.tar.gz
```

or the pretrained model `cpt-large`, which contains 24 encoder layers and 4 decoder layers:

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/nlp/CPT/cpt-large.tar.gz
tar -xzf cpt-base.tar.gz
```

Finally, run bash `train.sh` can train the model. Remember to change parameter `CPT_PRETRAIN_DIR` to correctly load the pretrained parameters.

```bash
sh train.sh
```

## Inference

Bash script `infer.sh` is used to test the pretrained model.

```bash
sh infer.sh
```

The default texts are `"订单退款成功了，为什么仍然要还花呗` and `"退款到了花呗，为什么还要还款"`, then the model will predict label `1` which means the two sentences are same in meanning. You can change them in `infer.sh`.