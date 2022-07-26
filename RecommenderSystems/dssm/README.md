# DSSM

[Deep Structured Semantic Model (DSSM)](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf)


## Directory description

```txt
.
├── dssm_train_eval.py  # OneFlow DSSM train/val/test scripts with OneEmbedding module
├── README.md           # Documentation
├── tools
│   ├── dssm_parquet.py # Read movielens data and export it as parquet data format
└── train_dssm.sh       # DSSM training shell script
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

A hands-on guide to train a DSSM model.

### Environment

1.   Install OneFlow by following the steps in [OneFlow Installation Guide](https://github.com/Oneflow-Inc/oneflow#install-oneflow) or use the command line below.

     ```shell
     python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu102
     ```

2.   Install all other dependencies listed below.

     ```json
     psutil
     petastorm
     ```

### Dataset


### Start Training by Oneflow

1.   Modify the **train_dssm.sh** as needed.

2.   train a DSSM model by `bash train_mmoe.sh`.
