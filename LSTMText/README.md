# TextBiLSTM

This repo is based on https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py.

## Training on IMDb dataset

### Preparing IMDb dataset

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/Imdb_ofrecord.tar.gz
tar zxf Imdb_ofrecord.tar.gz
```

### Train

```bash
bash train.sh
```

Note that the parameters of the provided pre-trained model is :

```bash
BATCH_SIZE=32
EPOCH=30
LEARNING_RATE=3e-4
SEQUENCE_LEN=128
EMBEDDING_DIM=100
NFC=128
HIDDEN_SIZE=256
```

## Inference on Single Text
```bash
bash infer.sh
```

The example text is a simple sentence `"It is awesome! It is nice. The director does a good job!"` (or `"The film is digusting!"`). You can change this text in `infer.sh`.