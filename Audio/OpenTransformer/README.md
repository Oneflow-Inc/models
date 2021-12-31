# OpenTransformer

This ia an Oneflow implementation of speech-transformer model for end-to-end speech recognition.

Our code is inspired by the Pytorch implementation [OpenTransformer](https://github.com/ZhengkunTian/OpenTransformer).


## Install
- Python3 (recommend Anaconda)
- Oneflow 0.6.0

## Workflow and Usage

### Data Preparation
```bash
cd egs/aishell
bash run.sh
```
### Train
```bash
bash train.sh
```
### Average the last N epochs
```bash
bash average.sh
```

### Eval
```bash
bash eval.sh
```
## Function

- Speech Transformer / Conformer

- Label Smoothing

- Tie Weights of Embedding with output softmax layer

- Data Augmentation([SpecAugument](https://arxiv.org/abs/1904.08779))

- Extract Fbank features in a online fashion

- Read the feature with the kaldi or espnet format!

- Batch Beam Search with Length Penalty

- Multiple Optimizers and Schedulers

- Multiple Activation Functions in FFN

- LM Shollow Fusion


## Experiments
Our Model can achieve a CER of 7.5% without CMVN, any external LM and joint-CTC training on [AISHELL-1](http://www.openslr.org/33/). The pretrained model can be obatained from [Speech_Transformer_Model](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/audio/OpenTransfomer.zip).



## Acknowledge
OpenTransformer refer to [ESPNET](https://github.com/espnet/espnet).
