# Tansformer

This repo is based on https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py.

## Training on IMDb dataset

### Preparing IMDb dataset

```bash
wget
tar zxf
```

### Train

```bash
bash train.sh
```

### Pre-trained model

```bash
wget xxx
tar zxf transformer_model.tar.gz
```

Note that the parameters of the provided pre-trained model is :

```bash
SEQUENCE_LEN=128
VOCAB_SZ=100000
D_MODEL=512
DROPOUT=0.1
NHEAD=8
NUM_LAYERS=6
DIM_FF=1024
```

## Inference on Single Text
```bash
bash infer.sh
```

The example text is a simple sentence `"This film is too bad."`. You can change this text in `infer.sh`.