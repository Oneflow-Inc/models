# DSSM

[Deep Structured Semantic Model (DSSM)](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf) is a two-tower deep learning recall model, which is originially used to calculate the semantic similarity between queries and documents.  Then, this model is applied to the field of personalized advertising recommendation.


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

We use exactly the same default values as [the DeepFM_Criteo_x4_001 experiment](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/DeepFM/DeepFM_criteo_x4_001) in FuxiCTR. 

| Argument Name              | Argument Explanation                                         | Default Value       |
| -------------------------- | ------------------------------------------------------------ | ------------------- |
| data_dir                   | the data file directory                                      | *Required Argument* |
| num_train_samples          | the number of train samples                                  | *Required Argument* |
| num_val_samples            | the number of validation samples                             | *Required Argument* |
| num_test_samples           | the number of test samples                                   | *Required Argument* |
| model_load_dir             | model loading directory                                      | None                |
| model_save_dir             | model saving directory                                       | None                |
| save_best_model            | save best model or not                                       | False               |
| save_initial_model         | save initial model parameters or not                         | False               |
| save_model_after_each_eval | save model after each eval or not                            | False               |
| embedding_vec_size         | embedding vector size                                        | 10                  |
| user_dnn_units             | user dnn hidden units number                                 | 400, 400, 400       |
| item_dnn_units             | item dnn hidden units number                                 | 400, 400, 400       |
| net_dropout                | number of minibatch training interations                     | 0.2                 |
| learning_rate              | initial learning rate                                        | 0.001               |
| batch_size                 | training/evaluation batch size                               | 4096                |
| train_batches              | the maximum number of training batches                       | 75000               |
| loss_print_interval        | interval of printing loss                                    | 100                 |
| patience                   | Number of epochs with no improvement after which learning rate will be reduced | 2                   |
| min_delta                  | threshold for measuring the new optimum, to only focus on significant changes | 1.0e-6              |
| table_size_array           | embedding table size array for sparse fields                 | *Required Argument* |
| persistent_path            | path for persistent kv store of embedding                    | *Required Argument* |
| store_type                 | OneEmbeddig persistent kv store type: `device_mem`, `cached_host_mem` or `cached_ssd` | `cached_host_mem`   |
| cache_memory_budget_mb     | size of cache memory budget on each device in megabytes when `store_type` is `cached_host_mem` or `cached_ssd` | 1024                |
| amp                        | enable Automatic Mixed Precision(AMP) training or not        | False               |
| loss_scale_policy          | loss scale policy for AMP training: `static` or `dynamic`    | `static`            |
| disable_early_stop         | disable early stop or not                                    | False               |

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