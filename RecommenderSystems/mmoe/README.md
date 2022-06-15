# MMoE

[Multi-gate Mixture-of-Experts (MMoE)](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007) adapts the Mixture-of- Experts (MoE) structure to multi-task learning by sharing the expert submodels across all tasks, while also having a gating network trained to optimize each task. Its model structure is as follows. Based on this structure, this project uses OneFlow distributed deep learning framework to realize training the model in graph mode on the Criteo data set.
<p align='center'>
  <img width="539" alt="mmoe" src="https://user-images.githubusercontent.com/46690197/172789523-443b5df5-432a-4b96-99ba-7712d46a81ed.png">
</p>

## Directory description

```txt
.
├── mmoe_train_eval.py  # OneFlow DeepFM train/val/test scripts with OneEmbedding module
├── README.md           # Documentation
├── tools
│   ├── mmoe_parquet.py # Read census-income data and export it as parquet data format
└── train_mmoe.sh       # MMoE training shell script
```

## Arguments description

| Argument Name              | Argument Explanation                                         | Default Value       |
| -------------------------- | ------------------------------------------------------------ | ------------------- |
| data_dir                   | the data file directory                                      | *Required Argument* |
| num_train_samples          | the number of train samples                                  | *Required Argument* |
| num_test_samples           | the number of test samples                                   | *Required Argument* |
| model_load_dir             | model loading directory                                      | None                |
| model_save_dir             | model saving directory                                       | None                |
| save_initial_model         | save initial model parameters or not                         | False               |
| save_model_after_each_eval | save model after each eval or not                            | False               |
| num_experts                | the number of experts                                        | 3                   |
| num_tasks                  | the number of tasks                                          | 2                   |
| embedding_vec_size         | embedding vector size                                        | 16                  |
| expert_dnn                 | expert dnn hidden units number                               | [256, 128]          |
| gate_dnn                   | gate dnn hidden units number                                 | []                  |
| tower_dnn                  | tower dnn hidden units number                                | []                  |
| net_dropout                | net dropout rate                                             | 0.0                 |
| learning_rate              | initial learning rate                                        | 0.001               |
| batch_size                 | training/evaluation batch size                               | 256                 |
| train_batches              | the maximum number of training batches                       | 16000               |
| loss_print_interval        | interval of printing loss                                    | 100                 |
| table_size_array           | embedding table size array for sparse fields                 | *Required Argument* |
| persistent_path            | path for persistent kv store of embedding                    | *Required Argument* |
| store_type                 | OneEmbeddig persistent kv store type: `device_mem`, `cached_host_mem` or `cached_ssd` | `cached_host_mem`   |
| cache_memory_budget_mb     | size of cache memory budget on each device in megabytes when `store_type` is `cached_host_mem` or `cached_ssd` | 1024                |
| amp                        | enable Automatic Mixed Precision(AMP) training or not        | False               |
| loss_scale_policy          | loss scale policy for AMP training: `static` or `dynamic`    | `static`            |


## Getting Started

A hands-on guide to train a MMoe model.

### Environment

1.   Install OneFlow by following the steps in [OneFlow Installation Guide](https://github.com/Oneflow-Inc/oneflow#install-oneflow) or use the command line below.

     ```shell
     python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu102
     ```

2.   Install all other dependencies listed below.

     ```json
     psutil
     petastorm
     pandas
     sklearn
     ```

### Dataset

### Start Training by Oneflow

1.   Modify the **train_mmoe.sh** as needed.

     ```shell
     #!/bin/bash
     DEVICE_NUM_PER_NODE=1
     DATA_DIR=/path/to/mmoe_parquet
     PERSISTENT_PATH=/path/to/persistent
     MODEL_SAVE_DIR=/path/to/model/save/dir
     
     python3 -m oneflow.distributed.launch \
         --nproc_per_node $DEVICE_NUM_PER_NODE \
         --nnodes 1 \
         --node_rank 0 \
         --master_addr 127.0.0.1 \
         mmoe_train_eval.py \
           --data_dir $DATA_DIR \
           --persistent_path $PERSISTENT_PATH \
           --table_size_array "9, 52, 47, 17, 3, 24, 15, 5, 10, 2, 3, 6, 8, 6, 6, 51, 38, 8, 10, 9, 10, 3, 4, 5, 43, 43, 43, 5, 3" \
           --store_type 'cached_host_mem' \
           --cache_memory_budget_mb 1024 \
           --batch_size 256 \
           --train_batches 16000 \
           --loss_print_interval 100 \
           --learning_rate 0.001 \
           --embedding_vec_size 4 \
           --expert_dnn "256, 128" \
           --num_train_samples 199523 \
           --num_test_samples 99762 \
           --model_save_dir $MODEL_SAVE_DIR
     ```
     
2.   train a MMoE model by `bash train_mmoe.sh`.
