# RoBERTA
- This repo is based on https://github.com/huggingface/transformers/tree/master/src/transformers/models/roberta.
- The `RoBERTA` model with related functions and models are placed in directory `roberta/`, including files as follows:

```
|-- README.md
|-- requirements.txt
|-- roberta
|   |-- MNLI                         # please download from aliyun
|   |-- MNLIDataset.py
|   |-- SST-2                        # please download from aliyun
|   |-- SST2Dataset.py
|   |-- classifier_MNLI.py
|   |-- classifier_SST2.py
|   |-- config.py
|   |-- roberta-base-oneflow            # please download from aliyun
|   |-- infer_MNLI.py
|   |-- infer_MNLI.sh
|   |-- infer_SST2.py
|   |-- infer_SST2.sh
|   |-- models
|   |-- pretrain_model_MNLI          # please download from aliyun
|   |-- pretrain_model_SST-2         # please download from aliyun
|   |-- train_MNLI.py
|   |-- train_MNLI.sh
|   |-- train_SST2.py
|   |-- train_SST2.sh
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

- We apply our `RoBERTA` model to text classfication task using both `SST-2` and `MNLI` datasets for the purpose of testing our model. The corresponding bash scripts and models are under `/roberta` .

## Model 

We implemented our model with refernce to `transformers.RobertaModel`, which is composed of `Roberta`, `RobertaEncoder`, `RobertaLayer`, `RobertaPooler`, `RobertaOutput`, `RobertaIntermediate`, `RobertaAttention`, `RobertaSelfAttention`, `RobertaSelfOutput`, `RobertaEmbedding`.

## requirement

This project uses the lightly version of oneflow. You can use the following command to install.
CPU：
```bash
python3 -m pip install -f  https://staging.oneflow.info/branch/master/cpu  --pre oneflow
```
GPU：
```bash
python3 -m pip install -f  https://staging.oneflow.info/branch/master/cu112  --pre oneflow
```
You can install other dependencies using the following command.
```bash
pip install -r requirements.txt
```

## Task 1 - Text Classfication

### Dataset

We complete NLP-classification task based on SST-2 dataset in order to test our model.
```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/NLP/SST-2.tar.gz
tar zxf SST-2.tar.gz
```
### Pretrain model
- roberta-base  <br>
We complete this by using the weights from already_trained roberta model from transformers. 

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/nlp/roberta/roberta-base-oneflow.tar.gz
tar zxf roberta-base-oneflow.tar.gz
```
- pretrain_model_SST-2 <br>
```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/nlp/pretrain_model_SST-2.tar.gz
tar zxf pretrain_model_SST-2.tar.gz
```

### Traininig

The bash script `train_SST2.sh` will train our model on SST-2 dataset.

```bash
cd roberta
sh train_SST2.sh
```

The default parameters are displayed below. You can modify them to fit your own environment.

```bash
ATCH_SIZE=32
EPOCH=20
LEARNING_RATE=1e-5
PRETRAIN_DIR="./roberta-base-oneflow/weights"
KWARGS_PATH="./roberta-base-oneflow/parameters.json"
SAVE_DIR="./pretrain_model_SST-2"
TASK="SST-2"
```

### Inference

Script `infer_SST2.sh` can test the result of classification. We use text `"this is junk food cinema at its greasiest ."` as example.

```bash
cd roberta
sh infer_SST2.sh
```

The default parameters are displayed below:

```bash
PRETRAIN_DIR="./roberta-base-oneflow/weights"
KWARGS_PATH="./roberta-base-oneflow/parameters.json"
MODEL_LOAD_DIR="./pretrain_model_SST-2"
TEXT="this is junk food cinema at its greasiest ."  
TASK="SST-2"
```

## Task 2 - Multi-Genre Natural Language Inference Task

### Dataset

We complete Multi-Genre Natural Language Inference task based on MNLI dataset in order to test our model.
```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/roberta/MNLI.tar.gz
tar zxf MNLI.tar.gz
```
### Pertrain model
- roberta-base  <br>
We complete this by using the weights from already_trained roberta model from transformers.
```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/nlp/roberta/roberta-base-oneflow.tar.gz
tar zxf roberta-base-oneflow.tar.gz
```
- pretrain_model_MNLI <br>
```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/nlp/roberta/pretrain_model_MNLI.tar.gz
tar zxf pretrain_model_MNLI.tar.gz
```

### Traininig

The bash script `train_MNLI.sh` will train our model on SST-2 dataset.

```bash
cd roberta
sh train_MNLI.sh
```

The default parameters are displayed below. You can modify them to fit your own environment.

```bash
BATCH_SIZE=32
EPOCH=20
LEARNING_RATE=1e-5
PRETRAIN_DIR="./roberta-base-oneflow/weights"
KWARGS_PATH="./roberta-base-oneflow/parameters.json"
SAVE_DIR="./pretrain_model_MNLI"
TASK="MNLI"
```

### Inference

Script `infer_MNLI.sh` can test the result of classification. We use text `"The new rights are nice enough. Everyone really likes the newest benefits."  ` as example.

```bash
cd roberta
sh infer_MNLI.sh
```

The default parameters are displayed below.

```bash
PRETRAIN_DIR="./roberta-base-oneflow/weights"
KWARGS_PATH="./roberta-base-oneflow/parameters.json"
MODEL_LOAD_DIR="./pretrain_model_MNLI"
TEXT="The new rights are nice enough. Everyone really likes the newest benefits."  
TASK="MNLI"
```