# BERT

Training BERT on [Wikipedia](https://dumps.wikimedia.org/) and [BookCorpus](http://yknzhu.wixsite.com/mbweb) for Pre-training and [SQuAD](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/squad_dataset_tools.tgz) for Fine-tuning using [OneFlow](https://github.com/Oneflow-Inc/oneflow#install-with-pip-package)

> BERT 2018 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
> Paper URL : https://arxiv.org/abs/1810.04805

## Introduction

Google AI's BERT paper shows the amazing result on various NLP task (new 17 NLP tasks SOTA), 
including outperform the human F1 score on SQuAD v1.1 QA task. 
This paper proved that Transformer(self-attention) based encoder can be powerfully used as 
alternative of previous language model with proper language model training method. 
And more importantly, they showed us that this pre-trained language model can be transfer 
into any NLP task without making task specific model architecture.


## Quickstart

### 0. Prepare Training Data

Before pre-training or fine-tuning, you need to prepare corresponding datasets as following instructions, and the dataset format is OFRecord.

- Pre-training dataset: The full dataset is consist of [Wikipedia](https://dumps.wikimedia.org/) and [BookCorpus](http://yknzhu.wixsite.com/mbweb), and it's about 200G. You can also try a [sample data](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/wiki_ofrecord_seq_len_128_example.tgz) for testing.
- SQuAD dataset: The full dataset and tools can be donwload in [here](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/squad_dataset_tools.tgz), extracted directory includes the following files:

```shell
squad_dataset_tools
├── ofrecord 
├── dev-v1.1.json  
├── dev-v2.0.json  
├── train-v1.1.json  
├── train-v2.0.json
├── evaluate-v1.1.py  
├── evaluate-v2.0.py  

```

### 1. Train on Single Device

OneFlow supports execution in eager mode or graph mode.

**Eager Execution**


```shell
export PYTHONUNBUFFERED=1
python3 run_eager_pretraining.py \
    --train-batch-size 32 \
    --lr 0.001 \
    --num_hidden_layers 12 \
    --num_attention_heads 12 \
    --max_position_embeddings 512 \
    --seq_length 128 \
    --vocab_size 30522 \
    --type_vocab_size 2 \
    --attention_probs_dropout_prob 0.1 \
    --hidden_dropout_prob 0.1 \
    --max_predictions_per_seq 20 \
    --ofrecord_path /dataset/bert/of_wiki_seq_len_128 \
    2>&1 | tee bert_eager_pretrain.log
```

**Graph Execution**

```shell
export PYTHONUNBUFFERED=1
python3 run_pretraining.py \
    --train-batch-size 32 \
    --lr 0.001 \
    --use_consistent \
    --num_hidden_layers 12 \
    --num_attention_heads 12 \
    --max_position_embeddings 512 \
    --seq_length 128 \
    --vocab_size 30522 \
    --type_vocab_size 2 \
    --attention_probs_dropout_prob 0.1 \
    --hidden_dropout_prob 0.1 \
    --max_predictions_per_seq 20 \
    --ofrecord_path /dataset/bert/of_wiki_seq_len_128 \
    2>&1 | tee bert_graph_pretrain.log
```

### 2. Distributed Training

Oneflow launches the distributed training the same way as Pytorch. For more details, please refer to the docs: oneflow.readthedocs.io/en/master/distributed.html?highlight=distributed#oneflow-distributed

But for known, we have only tested the distributed training on single node with 4 devices.

**Notice**: you need to `unset http_proxy` when distributed training.
```
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```

**Eager Execution**

```shell
export PYTHONUNBUFFERED=1
python3 -m oneflow.distributed.launch \
    --nproc_per_node 4 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port 12789 \
    run_eager_pretraining.py \
    --train-batch-size 16 \
    --lr 0.001 \
    --use_ddp \
    --num_hidden_layers 12 \
    --num_attention_heads 12 \
    --max_position_embeddings 512 \
    --seq_length 128 \
    --vocab_size 30522 \
    --type_vocab_size 2 \
    --attention_probs_dropout_prob 0.1 \
    --hidden_dropout_prob 0.1 \
    --max_predictions_per_seq 20 \
    --ofrecord_path /dataset/bert/of_wiki_seq_len_128 2>&1 | tee bert_eager_ddp_pretrain.log
```

**Graph Execution**

```shell
export PYTHONUNBUFFERED=1
python3 -m oneflow.distributed.launch \
    --nproc_per_node 4 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port 17789 \
    run_pretraining.py \
    --train-batch-size 32 \
    --lr 0.001 \
    --use_consistent \
    --use_fp16 \
    --num_hidden_layers 12 \
    --num_attention_heads 12 \
    --max_position_embeddings 512 \
    --seq_length 128 \
    --vocab_size 30522 \
    --type_vocab_size 2 \
    --attention_probs_dropout_prob 0.1 \
    --hidden_dropout_prob 0.1 \
    --max_predictions_per_seq 20 \
    --ofrecord_path /dataset/bert/of_wiki_seq_len_128 2>&1 | tee bert_graph_consistent_pretrain.log
```

**Explaination of command parameters for pretrain bert script**

```
--ofrecord_path                 Path to ofrecord dataset
--train-dataset-size            dataset size of ofrecord
--train-data-part               data part num of ofrecord
--train-batch-size              Training batch size
--val-batch-size                Validation batch size
--train-global-batch-size       train batch size
--val-global-batch-size         validation batch size
--num_hidden_layers             Number of layers
--num_attention_heads           Number of attention heads
--max_position_embeddings       Max position length of bert position embedding
--seq_length                    Maximum sequence len
--vocab_size                    Total number of vocab
--type_vocab_size               Number class of sentence classification
--attention_probs_dropout_prob  Attention dropout ratio
--hidden_dropout_prob           Hidden layer dropout ratio
--max_predictions_per_seq       Max prediction length
--epochs                        Number of epochs
--with-cuda                     Training with CUDA: true, or false
--cuda_devices                  CUDA device ids
--optim_name                    optimizer name
--lr                            Learning rate of adam
--weight_decay                  Weight_decay of adam
--warmup_proportion             Warmup propotion to total steps
--loss_print_every_n_iters      Interval of training loss printing
--val_print_every_n_iters       Interval of evaluation printing
--checkpoint_path               Path to model saving
--use_fp16                      Whether to use use fp16
--grad-acc-steps                Steps for gradient accumulation
--nccl-fusion-threshold-mb      NCCL fusion threshold megabytes, set to 0 to compatible 
                                with previous version of OneFlow.

--nccl-fusion-max-ops           Maximum number of ops of NCCL fusion, set to 0 
                                to compatible with previous version of OneFlow.

--use_ddp                       Whether to use use fp16
--use_consistent                Whether to use use consistent
--metric-local                  get local metric result
```

Notice: You can find some example scripts in `examples` for reference.

### 3. Fine-tune on SQuAD

After pre-training, you can run bert fine-tune on SQuAD with the following command. We just followed the original paper 
and run fine-tune with 1 GPU. You can also find the script in `example/run_squad.sh`.

```
# pretrained model dir
PRETRAINED_MODEL=snapshot_tf_for_graph_snapshot

# squad ofrecord dataset dir
DATA_ROOT=/dataset/bert/squad/ofrecord

# `vocab.txt` dir
REF_ROOT_DIR=/dataset/bert/uncased_L-12_H-768_A-12

# `evaluate-v*.py` and `dev-v*.json` dir
SQUAD_TOOL_DIR=/dataset/bert/squad

db_version=${1:-"v1.1"}
if [ $db_version = "v1.1" ]; then
  train_example_num=88614
  eval_example_num=10833
  version_2_with_negative="False"
elif [ $db_version = "v2.0" ]; then
  train_example_num=131944
  eval_example_num=12232
  version_2_with_negative="True"
else
  echo "db_version must be 'v1.1' or 'v2.0'"
  exit
fi

train_data_dir=$DATA_ROOT/train-$db_version
eval_data_dir=$DATA_ROOT/dev-$db_version
LOGFILE=./bert_fp_finetuning.log

# finetune and eval SQuAD, 
# `predictions.json` will be saved to folder `./squad_output`
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
python3 run_squad.py \
  --model=SQuAD \
  --do_train=True \
  --do_eval=True \
  --gpu_num_per_node=1 \
  --learning_rate=3e-5 \
  --batch_size_per_device=16 \
  --eval_batch_size_per_device=16 \
  --num_epoch=3 \
  --use_fp16 \
  --version_2_with_negative=$version_2_with_negative \
  --loss_print_every_n_iter=20 \
  --do_lower_case=True \
  --seq_length=384 \
  --num_hidden_layers=12 \
  --num_attention_heads=12 \
  --max_position_embeddings=512 \
  --type_vocab_size=2 \
  --vocab_size=30522 \
  --attention_probs_dropout_prob=0.1 \
  --hidden_dropout_prob=0.1 \
  --hidden_size_per_head=64 \
  --train_data_dir=$train_data_dir \
  --train_example_num=$train_example_num \
  --eval_data_dir=$eval_data_dir \
  --eval_example_num=$eval_example_num \
  --log_dir=./log \
  --model_load_dir=${PRETRAINED_MODEL} \
  --save_last_snapshot=True \
  --model_save_dir=./squad_snapshots \
  --vocab_file=$REF_ROOT_DIR/vocab.txt \
  --predict_file=$SQUAD_TOOL_DIR/dev-${db_version}.json \
  --output_dir=./squad_output 

# evaluate predictions.json to get metrics
python3 $SQUAD_TOOL_DIR/evaluate-${db_version}.py \
  $SQUAD_TOOL_DIR/dev-${db_version}.json \
  ./squad_output/predictions.json

```

After fine-tuning, you can get the evaluation results as follow
```
{"exact_match": 82.57332071901608, "f1": 89.63355726606137}
```

### 4. Inference 

You can refer to this [link](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/master/LanguageModeling/BERT) for 
converting TensorFlow model to OneFlow, then use it for inference.

We just provide a simple inference script in `example/infer.sh`, you can also run inference with following command

```
MODEL_PATH="your convert model"

python3 run_infer.py \
  --use_lazy_model \
  --model_path $MODEL_PATH
```

## Utils

### 1. Lazy Mode Loading
Bert lazy model can be loaded in `nn.Module` using the function `load_params_from_lazy` from `utils/compare_lazy_outputs.py`.

### 2. Draw Loss Curve
You can make use of `draw_loss_curve.py` to draw loss curve for aligning different train mode.
