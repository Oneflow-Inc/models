# Deep&Cross
 [Deep & Cross Network](https://dl.acm.org/doi/10.1145/3124749.3124754) (DCN) can not only keep the advantages of DNN model, but also learn specific bounded feature crossover more effectively. In particular, DCN can explicitly learn cross features for each layer without the need for manual feature engineering, and the increased algorithm complexity is almost negligible compared with DNN model.
 ![DCN](https://user-images.githubusercontent.com/80230303/159417248-1975736f-3de8-4972-84e3-2f0f346cbc1a.png)


Oneflow API is compatible to Pytorch, so only minor modification in codes then we can apply the Pytorch implemented modules to Oneflow. Therefore, we adopted some implementation from [FuxiCTR](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2), for example the `CrossInteractionLayer` was reused as a basic `CrossNet` hidden layer, and we make a recurrent loop of these hidden layers in `CrossNet` module to get high-degree interaction across features.

## Directory description
```
.
|-- tools
  |-- dcn_parquet.scala   # Read Criteo Kaggle data and export it as parquet data format
  |-- split_criteo.py         # Split criteo kaggle dataset to train\val\test csv files
  |-- launch_spark.sh        # Spark launching shell script
|-- dcn_train_eval.py       # OneFlow DCN train/val/test scripts with OneEmbedding module
|-- train.sh                # DCN training shell script
|-- requirements.txt         # python package configuration file
└── README.md                # Documentation
```


## Arguments description
We use exactly the same default values as the [DCN_criteo_x4_001](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/DCN/DCN_criteo_x4_001) experiment in FuxiCTR.
|Argument Name|Argument Explanation|Default Value|
|-----|---|------|
|data_dir|the data file directory|*Required Argument*|
|num_train_samples|the number of training samples|36672493|
|num_valid_samples|the number of validation samples|4584062|
|num_test_samples|the number of test samples|4584062|
|shard_seed|seed for shuffling parquet data|2022|
|model_load_dir|model loading directory|None|
|model_save_dir|model saving directory|None|
|save_best_model|save best model or not|False|
|save_initial_model|save initial model parameters or not|False|
|save_model_after_each_eval|save model or not after each evaluation|False|
|embedding_vec_size|embedding vector dimention size|128|
|batch_norm|batch norm used in DNN|False|
|dnn_hidden_units|hidden units list of DNN|"1000,1000,1000,1000,1000"|
|crossing_layers|layer number of Crossnet|3|
|net_dropout|dropout rate of DNN|0.2|
|embedding_regularizer|rate of embedding layer regularizer|None|
|net_regularizer|rate of Crossnet and DNN layer regularizer|None|
|disable_early_stop|disable early stop or not|False|
|patience|waiting epoch of ealy stopping|2|
|min_delta|minimal delta of metric Monitor|1.0e-6|
|lr_factor|learning rate decay factor|0.1|
|min_lr|minimal learning rate|1.0e-6|
|learning_rate|learning rate|0.001|
|size_factor|size factor of OneEmbedding|3|
|valid_batch_size|valid batch size|10000|
|valid_batches|number of valid batches|1000|
|test_batch_size|test batch size|10000|
|test_batches|number of test batches|1000|
|train_batch_size|train batch size|10000|
|train_batches|number of train batches|15000|
|loss_print_interval|training loss print interval|100|
|train_batch_size|training batch size|55296|
|train_batches|number of minibatch training interations|75000|
|table_size_array|table size array for sparse fields|*Required Argument*|
|persistent_path|path for OneEmbedding persistent kv store|*Required Argument*|
|store_type|OneEmbeddig persistent kv store type: `device_mem`, `cached_host_mem` or `cached_ssd` |cached_ssd|
|cache_memory_budget_mb|size of cache memory budget on each device in megabytes when `store_type` is `cached_host_mem` or `cached_ssd`|8192|
|amp|enable Automatic Mixed Precision(AMP) training|False|
|loss_scale_policy|loss scale policy for AMP training: `static` or `dynamic`|static|

#### Early Stop Schema

The model is evaluated at the end of every epoch. At the end of each epoch, if the early stopping criterion is met, the training process will be stopped. 

The monitor used for the early stop is `val_auc - val_log_loss`. The mode of the early stop is `max`. You could tune `patience` and `min_delta` as needed.

If you want to disable early stopping, simply add `--disable_early_stop` in the [train.sh](https://github.com/Oneflow-Inc/models/blob/criteo_dcn/RecommenderSystems/dcn/train.sh).


## Getting started
If you'd like to quickly train a OneFlow DCN model, please follow steps below:
### Installing OneFlow and Dependencies
1. To install nightly release of OneFlow with CUDA 11.5 support:
```
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu115
```
For more information how to install Oneflow, please refer to [Oneflow Installation Tutorial](
https://github.com/Oneflow-Inc/oneflow#install-oneflow).

2. Please check `requirements.txt` to install dependencies manually or execute:
```bash
python3 -m pip install -r requirements.txt
```
### Dataset
The Criteo dataset is from [2014-kaggle-display-advertising-challenge-dataset](https://www.kaggle.com/competitions/criteo-display-ad-challenge/overview), considered the original download link is invalid, click [here](https://www.kaggle.com/datasets/mrkmakr/criteo-dataset) to donwload if you would.

Each sample contains:
- Label - Target variable that indicates if an ad was clicked (1) or not (0).
- I1-I13 - A total of 13 columns of integer features (mostly count features).
- C1-C26 - A total of 26 columns of categorical features. The values of these features have been hashed onto 32 bits for anonymization purposes.


1. Download the [Criteo Kaggle dataset](https://www.kaggle.com/c/criteo-display-ad-challenge) and then split it using split_criteo_kaggle.py.

2. launch a spark shell using [launch_spark.sh](https://github.com/Oneflow-Inc/models/blob/criteo_dcn/RecommenderSystems/dcn/tools/launch_spark.sh).

     -   Modify the SPARK_LOCAL_DIRS as needed

         ```shell
         export SPARK_LOCAL_DIRS=/path/to/your/spark/
         ```

     -   Run `bash launch_spark.sh`

3. load [dcn_parquet.scala](https://github.com/Oneflow-Inc/models/blob/criteo_dcn/RecommenderSystems/dcn/tools/dcn_parquet.scala) to your spark shell by `:load dcn_parquet.scala`.

4. call the `makeDCNDataset(srcDir: String, dstDir:String)` function to generate the dataset.

     ```shell
     makeDCNDataset("/path/to/your/src_dir", "/path/to/your/dst_dir")
     ```

     After generating parquet dataset, dataset information will also be printed. It contains the information about the number of samples and table size array, which is needed when training.

     ```txt
     train samples = 36672493                                                             
     validation samples = 4584062
     test samples = 4584062                                                               
     table size array: 
     649,9364,14746,490,476707,11618,4142,1373,7275,13,169,407,1376,1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572
     ```


## Start training by Oneflow
Following command will launch 8 oneflow DCN training and evaluation processes on a node with 8 GPU devices, by specify `data_dir` for data input and `persistent_path` for OneEmbedding persistent store path.

`table_size_array` is close related to sparse features of data input. each sparse field such as `C1` or other `C*` field in criteo dataset corresponds to a embedding table and has its own capacity of unique feature ids, this capacity is also called `number of rows` or `size of embedding table`, the embedding table will be initialized by this value. `table_size_array` holds all sparse fields' `size of embedding table`. `table_size_array` is also used to estimate capacity for OneEmbedding. 

```python
DEVICE_NUM_PER_NODE=8
DATA_DIR=your_path/criteo_parquet
PERSISTENT_PATH=your_path/persistent1
MODEL_SAVE_DIR=your_path/model_save_dir

python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    dcn_train_eval.py \
      --data_dir $DATA_DIR \
      --model_save_dir $MODEL_SAVE_DIR \
      --persistent_path $PERSISTENT_PATH \
      --table_size_array "649,9364,14746,490,476707,11618,4142,1373,7275,13,169,407,1376,1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572" \
      --store_type 'cached_host_mem' \
      --cache_memory_budget_mb 2048 \
      --dnn_hidden_units "1000, 1000, 1000, 1000, 1000" \
      --crossing_layers 4 \
      --embedding_vec_size 16 

```

You could modified it in [train.sh](https://github.com/Oneflow-Inc/models/blob/criteo_dcn/RecommenderSystems/dcn/train.sh), and then quickly run by 

`
bash train.sh
`




