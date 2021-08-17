# RoBERTA
- This repo is based on https://github.com/huggingface/transformers/tree/master/src/transformers/models/roberta.
- The `RoBERTA` model with related functions and models are placed in directory `roberta/`, including files as follows:
```
|-- README.md
|-- roberta
|   |-- README.md
|   |-- SST-2                       # please download from aliyun
|   |-- classifier_flow.py
|   |-- infer.py
|   |-- infer.sh
|   |-- models
|   |-- myDataset_flow.py
|   |-- pretrain_model_SST-2        # please download from aliyun
|   |-- random_test.py              # to test different roberta model
|   |-- read_data.py
|   |-- requirements.txt
|   |-- roberta_pretrain_oneflow    # please download from aliyun
|   |-- train_flow.py
|   |-- train_flow.sh
|   `-- weights_transform
`-- tokenizer
    |-- Conversation.py
    |-- GPT2Tokenizer.py
    |-- README.md
    |-- RobertaTokenizer.py
    |-- __init__.py
    |-- file_utils.py
    |-- requirements.txt
    |-- tokenization_utils.py
    |-- tokenization_utils_base.py
    `-- utils
```

- We apply our `RoBERTA` model to text classfication task using `SST-2` dataset for the purpose of testing our model. The corresponding bash scripts and models are under `/roberta` .

## Model 

We implemented our model with refernce to `transformers.RobertaModel`, which is composed of `Roberta`, `RobertaEncoder`, `RobertaLayer`, `RobertaPooler`, `RobertaOutput`, `RobertaIntermediate`, `RobertaAttention`, `RobertaSelfAttention`, `RobertaSelfOutput`, `RobertaEmbedding`.

There are some differences between `transformers.RobertaModel` and ours:

- `oneflow.nn.LayerNorm` suffers from some problems so that when training with GPU, the model won't converge properly. So we put it in `dev_ops.py`
- `oneflow.einsum` and `oneflow.cumsum` has not been implemented yet, So in `roberta_utils.py` we write some new functions to ensure that our model can run correctly.

## Task 1 - Text Classfication

### Dataset

We complete NLP-classification task based on SST-2 dataset in order to test our model.
```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/NLP/SST-2.tar.gz
tar zxf SST-2.tar.gz
```
### Pertrain model
- roberta_pretrain_oneflow  <br>
We complete this by moving the weights from already_trained roberta model from transformers. For more details, see weights_transform. 
```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/nlp/roberta_pretrain_oneflow.tar.gz
tar zxf roberta_pretrain_oneflow.tar.gz
```
- pretrain_model_SST-2 <br>
```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/nlp/pretrain_model_SST-2.tar.gz
tar zxf pretrain_model_SST-2.tar.gz
```

### Traininig

The bash script `train_flow.sh` will train our model on SST-2 dataset.

```bash
cd roberta
sh train_flow.sh
```

The default parameters are displayed below. You can modify them to fit your own environment.

```bash
BATCH_SIZE=32
EPOCH=10
LEARNING_RATE=1e-5
PRETRAIN_DIR="./roberta_pretrain_oneflow/roberta-base"
KWARGS_PATH="./roberta_pretrain_oneflow/roberta-base/roberta-base.json"
SAVE_DIR="./pretrain_model_SST-2"
```

### Inference

Script `infer.sh` can test the result of classification. We use text `"this is junk food cinema at its greasiest ."` as example.

```bash
cd roberta
sh infer.sh
```

The default parameters are displayed below.

```bash
PRETRAIN_DIR="./roberta_pretrain_oneflow/roberta-base"
KWARGS_PATH="./roberta_pretrain_oneflow/roberta-base/roberta-base.json"
MODEL_LOAD_DIR="./pretrain_model_SST-2"
TEXT="this is junk food cinema at its greasiest ." 
```