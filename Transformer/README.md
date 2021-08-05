# Transformer
- This repo is based on https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py.
- The `Transformer` model with related functions and models are placed in directory `transformer/`.
- We apply our `Transformer` model to text classfication task using `Imdb` dataset and odd number generation task to test our model. The corresponding bash scripts and models are under `imdb/` and `odd_numbers/`.

## Model

We implemented our model with reference to `torch.nn.modules.transformer`, which is composed of `Transformer`, `TransformerEncoder`, `TransformerDecoder`, `TransformerEncoderLayer`,
`TransformerDecoderLayer`.

Note that Modules in oneflow do not support `copy.deepcopy`, thus there are some differences when initializing these models between torch's Transformer and ours. What's more, we do not need paramerters `device` and `dtype` in oneflow.

Also, We found out that `oneflow.nn.LayerNorm` suffers from some problems so that when training with GPU, the model won't converge properly. And function `logical_or` haven't been implemented yet in oneflow. Therefore we put thems in `dev_ops.py` to ensure we can run our `Transformer` correctly. `dev_ops.py` will be removed when these problems are fixed.

## Task 1 - Text Classfication

### Dataset

We use Imdb dataset to test our model first.

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/Imdb_ofrecord.tar.gz
tar zxf Imdb_ofrecord.tar.gz
```

### Traininig

The bash script `imdb/train.sh` will train our model on imdb dataset.

```bash
cd imdb
sh train.sh
```

The default parameters are displayed below. You can modify them to fit your own environment.

```bash
BATCH_SIZE=32
EPOCH=1
LEARNING_RATE=0.0001

SEQUENCE_LEN=128
VOCAB_SZ=100000
D_MODEL=512
DROPOUT=0.1
NHEAD=8
NUM_LAYERS=4
DIM_FF=1024

IMDB_PATH="../../imdb"
LOAD_DIR="."
SAVE_DIR="best_model"
```

### Inference

Script `imdb/infer.sh` can test the result of classification. We use text `"It is awesome! It is nice. The director does a good job!"` as example.

```bash
cd imdb
sh infer.sh
```

The default parameters are displayed below.

```bash
SEQUENCE_LEN=128
VOCAB_SZ=100000
D_MODEL=512
DROPOUT=0.1
NHEAD=8
NUM_LAYERS=4
DIM_FF=1024
LOAD_DIR="best_model"
TEXT="It is awesome! It is nice. The director does a good job!"
```

## Task 2 - Generating Odd Numbers

This task can generate a sequence of odd numbers according to the input even numbers. For example, if we input `[2422, 2424, 2426]`, then we will get `[0, 2423, 2425, 2427]` as a result.

### Dataset

We generate the data by ourselves. Please read `odd_numbers/train_transformer_odd_numbers.py` for details.

### Training

You can use bash script `odd_numbers/train.sh` to train this model.

```bash
cd odd_numbers
sh train.sh
```

Note that the default parameters are following:

```bash
BATCH_SIZE=128
EPOCH=20
LEARNING_RATE=0.0001

VOCAB_SZ=10000
D_MODEL=512
DROPOUT=0.0
NHEAD=2
NUM_ENCODER_LAYERS=1
NUM_DECODER_LAYERS=1
DIM_FF=128

LOAD_DIR="."
SAVE_DIR="best_model"
```

### inference

Bash script `odd_numbers/infer.sh` is used to infer the trained model.

```bash
cd odd_numbers
sh infer.sh
```

The default parameters are set as below:

```bash
VOCAB_SZ=10000
D_MODEL=512
DROPOUT=0.0
NHEAD=2
NUM_ENCODER_LAYERS=1
NUM_DECODER_LAYERS=1
DIM_FF=128

LOAD_DIR="best_model"
INPUT_START=4386
```

The parameter `input_start` is the first number of the sequence input. If it is 4386, then the program will generate the sequence `[4386, 4388, 4390]` as input.