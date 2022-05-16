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

We use exactly the same default arguments as [the DeepFM_Criteo_x4_001 experiment](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/DeepFM/DeepFM_criteo_x4_001) in FuxiCTR. 

| Argument Name              | Argument Explanation                                         | Default Value            |
| -------------------------- | ------------------------------------------------------------ | ------------------------ |
| data_dir                   | the data file directory                                      | *Required Argument*      |
| num_train_samples          | the number of train samples                                  | *Required Argument*      |
| num_val_samples            | the number of validation samples                             | *Required Argument*      |
| num_test_samples           | the number of test samples                                   | *Required Argument*      |
| model_load_dir             | model loading directory                                      | None                     |
| model_save_dir             | model saving directory                                       | None                     |
| save_best_model            | save best model or not                                       | False                    |
| save_initial_model         | save initial model parameters or not                         | False                    |
| save_model_after_each_eval | save model after each eval or not                            | False                    |
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
| store_type                 | OneEmbeddig persistent kv store type: `device_mem`, `cached_host_mem` or `cached_ssd` | `cached_host_mem`        |
| cache_memory_budget_mb     | size of cache memory budget on each device in megabytes when `store_type` is `cached_host_mem` or `cached_ssd` | 1024                     |
| amp                        | enable Automatic Mixed Precision(AMP) training or not        | False                    |
| loss_scale_policy          | loss scale policy for AMP training: `static` or `dynamic`    | `static`                 |
| disable_early_stop         | disable early stop or not                                    | False                    |

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
     psutil: 5.9.0
     petastorm: 0.11.4
     pandas: 1.4.1
     pyspark: 3.2.1
     ```

### Dataset

**TODO**：make our own dataset

For now, we use the criteo_x4_001 dataset in FuxiCTR. We also use exactly the same data preprocessing steps as FuxiCTR by directly converting the preprocessed dataset to parquet format.

**Note**: 

According to [the DeepFM paper](https://arxiv.org/abs/1703.04247), we treat both categorical and continuous features as sparse features. 

>   χ may include cat- egorical fields (e.g., gender, location) and continuous fields (e.g., age). Each categorical field is represented as a vec- tor of one-hot encoding, and each continuous field is repre- sented as the value itself, or a vector of one-hot encoding af- ter discretization. 

Besides, for FuxiCTR [Criteo_x4_001](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x4_001), features with frequency < 10 are dropped. This is not mentioned in the original paper. We will run more experiments later to check if this step is necessary.

1.   Follow the steps in FuxiCTR [Criteo_x4_001](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x4_001) to download and split the dataset.
2.   Follow the steps in [FuxiCTR's DeepFM Criteo x4 001 experiment guide](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/DeepFM/DeepFM_criteo_x4_001) to train a DeepFM on Criteo_x4_001 dataset. After the experiment is done, a directory which contains five files, named `feature_encoder.plk`, `feature_map.json`, `train.h5`, `valid.h5`, and `test.h5` respectively, will be generated, which is the preprocessed dataset.
3.   Use [h5_to_parquet.py](https://github.com/Oneflow-Inc/models/blob/dev_deepfm/RecommenderSystems/deepfm/tools/h5_to_parquet.py) to convert it to parquet format.
     ```shell
     python h5_to_parquet.py \
          --input_dir=/path/to/preprocessed_dataset \
          --output_dir=/path/to/deepfm_parquet \
          --spark_tmp_dir=/path/to/spark_tmp_dir \
          --export_dataset_info
     ```
​	When generating parquet dataset, a README.md file will also be generated. It contains the information about the number of samples and table size array, which is needed when training.

     ```markdown
     ## number of examples:
     train: 36672493
     test: 4584062
     val: 4584062

     ## table size array
     table_size_array = [43, 98, 121, 41, 219, 112, 79, 68, 91, 5, 26, 36, 70, 1447, 554, 157461, 117683, 305, 17, 11878, 629, 3, 39504, 5128, 156729, 3175, 27, 11070, 149083, 10, 4542, 1996, 4, 154737, 17, 15, 52989, 81, 40882]
     ```

### Start Training by Oneflow

1.   Modify the [train_deepfm_criteo_x4.sh](https://github.com/Oneflow-Inc/models/blob/dev_deepfm/RecommenderSystems/deepfm/train_deepfm_criteo_x4.sh) as needed.

     ```shell
     #!/bin/bash
     DEVICE_NUM_PER_NODE=1
     DATA_DIR=/path/to/deepfm_parquet
     PERSISTENT_PATH=/path/to/persistent
     MODEL_SAVE_DIR=/path/to/model/save/dir

     python3 -m oneflow.distributed.launch \
     --nproc_per_node $DEVICE_NUM_PER_NODE \
     --nnodes 1 \
     --node_rank 0 \
     --master_addr 127.0.0.1 \
     deepfm_train_eval.py \
          --data_dir $DATA_DIR \
          --persistent_path $PERSISTENT_PATH \
          --table_size_array "43, 98, 121, 41, 219, 112, 79, 68, 91, 5, 26, 36, 70, 1447, 554, 157461, 117683, 305, 17, 11878, 629, 4, 39504, 5128, 156729, 3175, 27, 11070, 149083, 11, 4542, 1996, 4, 154737, 17, 16, 52989, 81, 40882" \
          --store_type 'cached_host_mem' \
          --cache_memory_budget_mb 1024 \
          --batch_size 10000 \
          --train_batches 75000 \
          --loss_print_interval 100 \
          --dnn "1000,1000,1000,1000,1000" \
          --net_dropout 0.2 \
          --learning_rate 0.001 \
          --embedding_vec_size 16 \
          --num_train_samples 36672493 \
          --num_val_samples 4584062 \
          --num_test_samples 4584062 \
          --model_save_dir $MODEL_SAVE_DIR \
          --save_best_model
     ```

2.   train a DeepFM model by `bash train_deepfm_criteo_x4.sh`.

## Performance

| Dataset       | Test LogLoss | Test AUC |
| ------------- | ------------ | -------- |
| Criteo_x4_001 | 0.808978     | 0.443073 |
