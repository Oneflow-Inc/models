# DeepFM

[DeepFM](https://arxiv.org/abs/1703.04247) is a Factorization-Machine based Neural Network for CTR prediction. Its model structure is as follows. Based on this structure, this project uses OneFlow distributed deep learning framework to realize training the model in graph mode on the Criteo data set.

<p align='center'>
  <img width="539" alt="Screen Shot 2022-04-01 at 4 45 22 PM" src="https://user-images.githubusercontent.com/46690197/161228714-ae9410bb-56db-46b0-8f0b-cb8becb6ee03.png">
</p>

## Directory description

```txt
.
├── deepfm_train_eval.py       # OneFlow DeepFM train/val/test scripts with OneEmbedding module
├── README.md                  # Documentation
├── tools
│   ├── criteo_parquet.py      # Read Criteo Kaggle data and export it as parquet data format
│   ├── h5_to_parquet.py       # Read .h5 data preprocessed by FuxiCTR and export it as parquet data format
├── train_deepfm_criteo_x4.sh  # DeepFM training shell script

```

## Arguments description

| Argument Name              | Argument Explanation                                         | Default Value            |
| -------------------------- | ------------------------------------------------------------ | ------------------------ |
| data_dir                   | the data file directory                                      | *Required Argument*      |
| num_train_samples          | the number of train samples                                  | *Required Argument*      |
| num_val_samples            | the number of validation samples                             | *Required Argument*      |
| num_test_samples           | the number of test samples                                   | *Required Argument*      |
| model_load_dir             | model loading directory                                      | None                     |
| model_save_dir             | model saving directory                                       | None                     |
| save_initial_model         | save initial model parameters or not                         | False                    |
| save_model_after_each_eval | save model after each eval or not                            | False                    |
| disable_fusedmlp           | disable fused MLP or not                                     | True                     |
| embedding_vec_size         | embedding vector size                                        | 16                       |
| dnn                        | dnn hidden units number                                      | 1000,1000,1000,1000,1000 |
| net_dropout                | number of minibatch training interations                     | 0.2                      |
| embedding_vec_size         | embedding vector size                                        | 16                       |
| learning_rate              | initial learning rate                                        | 0.001                    |
| batch_size                 | training/evaluation batch size                               | 10000                    |
| train_batches              | the maximum number of training batches                       | 75000                    |
| loss_print_interval        | interval of printing loss                                    | 100                      |
| patience                   | Number of epochs with no improvement after which learning rate will be reduced | 2                        |
| min_delta                  | threshold for measuring the new optimum, to only focus on significant changes | 1.0e-6                   |
| table_size_array           | embedding table size array for sparse fields                 | *Required Argument*      |
| persistent_path            | path for persistent kv store of embedding                    | *Required Argument*      |
| persistent_path_fm         | path for persistent kv store of embedding in FM              | *Required Argument*      |
| store_type                 | OneEmbeddig persistent kv store type: `device_mem`, `cached_host_mem` or `cached_ssd` | `cached_host_mem`        |
| cache_memory_budget_mb     | size of cache memory budget on each device in megabytes when `store_type` is `cached_host_mem` or `cached_ssd` | 1024                     |
| amp                        | enable Automatic Mixed Precision(AMP) training or not        | False                    |
| loss_scale_policy          | loss scale policy for AMP training: `static` or `dynamic`    | `static`                 |

## Getting Started

A hands-on guide to train a DeepFM model.

### Environment

1.   Install OneFlow by following the steps in [OneFlow Installation Guide](https://github.com/Oneflow-Inc/oneflow#install-oneflow) or use the command line below.

     ```shell
     python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu102
     ```

2.   Install all other dependencies listed below.

```json
CUDA: 10.2
python: 3.8.4
oneflow: 0.8.0
numpy: 1.19.2
scipy: 1.7.3
sklearn: 1.0.2
psutil: 5.9.0
petastorm: 0.11.4
pandas: 1.4.1
pyspark: 3.2.1
```

### Dataset

**TODO**：make our own dataset

For now, we use the criteo_x4_001 dataset in FuxiCTR.

1.   Follow the steps in FuxiCTR [Criteo_x4_001](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x4_001) to download and split the dataset.

2.   Use [h5_to_parquet.py](https://github.com/Oneflow-Inc/models/blob/dev_deepfm/RecommenderSystems/deepfm/tools/h5_to_parquet.py) to convert it to parquet format.

     ```shell
     python h5_to_parquet.py \
         --input_dir=/path/to/dataset \
         --output_dir=/path/to/deepfm_parquet \
         --spark_tmp_dir=PATH TO PYSPARK TMP DIRECTORY \
         --export_dataset_info
     ```

### Start Training by Oneflow

1.   Modify the [train_deepfm_criteo_x4.sh](https://github.com/Oneflow-Inc/models/blob/dev_deepfm/RecommenderSystems/deepfm/train_deepfm_criteo_x4.sh) as needed.
2.   train a DeepFM model by `bash train_deepfm_criteo_x4.sh`.

## Performance

