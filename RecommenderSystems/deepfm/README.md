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
│   ├── deepfm_parquet.scala   # Read Criteo Kaggle data and export it as parquet data format
│   └── launch_spark.sh        # Spark launching shell script
│   └── split_criteo_kaggle.py # Split criteo kaggle dataset to train\val\test set
├── train_deepfm.sh            # DeepFM training shell script
```

## Arguments description

We use exactly the same default values as [the DeepFM_Criteo_x4_001 experiment](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/DeepFM/DeepFM_criteo_x4_001) in FuxiCTR. 

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
| disable_fusedmlp           | disable fused mlp or not                                     | False                    |
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

#### Early Stop Schema

The model is evaluated at the end of every epoch. At the end of each epoch, if the early stopping criterion is met, the training process will be stopped. 

The monitor used for the early stop is `val_auc - val_log_loss`. The mode of the early stop is `max`. You could tune `patience` and `min_delta` as needed.

If you want to disable early stopping, simply add `--disable_early_stop` in the [train_deepfm.sh](https://github.com/Oneflow-Inc/models/blob/dev_deepfm_multicol_oneemb/RecommenderSystems/deepfm/train_deepfm.sh).

## Getting Started

A hands-on guide to train a DeepFM model.

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

**Note**: 

According to [the DeepFM paper](https://arxiv.org/abs/1703.04247), we treat both categorical and continuous features as sparse features. 

>   χ may include categorical fields (e.g., gender, location) and continuous fields (e.g., age). Each categorical field is represented as a vec- tor of one-hot encoding, and each continuous field is repre- sented as the value itself, or a vector of one-hot encoding after discretization. 

1.   Download the [Criteo Kaggle dataset](https://www.kaggle.com/c/criteo-display-ad-challenge) and then split it using [split_criteo_kaggle.py](https://github.com/Oneflow-Inc/models/blob/dev_deepfm_multicol_oneemb/RecommenderSystems/deepfm/tools/split_criteo_kaggle.py).

     Note: Same as [the DeepFM_Criteo_x4_001 experiment](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/DeepFM/DeepFM_criteo_x4_001) in FuxiCTR, only train.txt is used. Also, the dataset is randomly spllitted into 8:1:1 as training set, validation set and test set. The dataset is splitted using StratifiedKFold in sklearn.

     ```shell
     python3 split_criteo_kaggle.py --input_dir=/path/to/your/criteo_kaggle --output_dir=/path/to/your/output/dir
     ```

2.   Download spark from https://spark.apache.org/downloads.html and then uncompress the tar file into the directory where you want to install Spark. Ensure the `SPARK_HOME` environment variable points to the directory where the spark is.

3.   launch a spark shell using [launch_spark.sh](https://github.com/Oneflow-Inc/models/blob/dev_deepfm_multicol_oneemb/RecommenderSystems/deepfm/tools/launch_spark.sh).

     -   Modify the SPARK_LOCAL_DIRS as needed

         ```shell
         export SPARK_LOCAL_DIRS=/path/to/your/spark/
         ```

     -   Run `bash launch_spark.sh`

4.   load [deepfm_parquet.scala](https://github.com/Oneflow-Inc/models/blob/dev_deepfm_multicol_oneemb/RecommenderSystems/deepfm/tools/deepfm_parquet.scala) to your spark shell by `:load deepfm_parquet.scala`.

5.   call the `makeDeepfmDataset(srcDir: String, dstDir:String)` function to generate the dataset.

     ```shell
     makeDeepfmDataset("/path/to/your/src_dir", "/path/to/your/dst_dir")
     ```

     After generating parquet dataset, dataset information will also be printed. It contains the information about the number of samples and table size array, which is needed when training.

     ```txt
     train samples = 36672493                                                             
     validation samples = 4584062
     test samples = 4584062                                                               
     table size array: 
     649,9364,14746,490,476707,11618,4142,1373,7275,13,169,407,1376
     1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572
     ```

### Start Training by Oneflow

1.   Modify the [train_deepfm.sh](https://github.com/Oneflow-Inc/models/blob/dev_deepfm_multicol_oneemb/RecommenderSystems/deepfm/train_deepfm.sh) as needed.

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
          --table_size_array "649,9364,14746,490,476707,11618,4142,1373,7275,13,169,407,1376,1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572" \
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

2.   train a DeepFM model by `bash train_deepfm.sh`.
