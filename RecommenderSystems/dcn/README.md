# Deep&Cross
 [Deep & Cross Network](https://dl.acm.org/doi/10.1145/3124749.3124754) (DCN) can not only keep the advantages of DNN model, but also learn specific bounded feature crossover more effectively. In particular, DCN can explicitly learn cross features for each layer without the need for manual feature engineering, and the increased algorithm complexity is almost negligible compared with DNN model.


## Directory description
```
.
|-- tools
  |-- dataset_config.yaml   # dataset config file
  |-- split_criteo.py      # split Criteo txt file to csv files
  |-- make_criteo_parquet.py # make Criteo parquet data from csv files
|-- dcn_train_eval.py       # OneFlow DCN training and evaluation scripts with OneEmbedding module
|-- train.sh                # command to train DCN
|-- requirements.txt         # python package configuration file
└── README.md                # Documentation
```


## Arguments description
|Argument Name|Argument Explanation|Default Value|
|-----|---|------|
|data_dir|the data file directory|*Required Argument*|
|num_train_samples|the number of training samples|36672493|
|num_valid_samples|the number of validation samples|4584062|
|num_test_samples|the number of test samples|4584062|
|shard_seed|seed for shuffling parquet data|2022|
|model_load_dir|model loading directory|None|
|model_save_dir|model saving directory|None|
|save_initial_model|save initial model parameters or not|False|
|save_model_after_each_eval|save model or not after each evaluation|False|
|embedding_vec_size|embedding vector dimention size|128|
|batch_norm|batch norm used in DNN|False|
|dnn_hidden_units|hidden units list of DNN|"1000,1000,1000,1000,1000"|
|crossing_layers|layer number of Crossnet|3|
|net_dropout|dropout rate of DNN|0.2|
|embedding_regularizer|rate of embedding layer regularizer|None|
|net_regularizer|rate of Crossnet and DNN layer regularizer|None|
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
|loss_print_interval|training loss print interval|1000|
|train_batch_size|training batch size|55296|
|train_batches|number of minibatch training interations|75000|
|table_size_array|table size array for sparse fields|*Required Argument*|
|persistent_path|path for OneEmbedding persistent kv store|*Required Argument*|
|store_type|OneEmbeddig persistent kv store type: `device_mem`, `cached_host_mem` or `cached_ssd` |cached_ssd|
|cache_memory_budget_mb|size of cache memory budget on each device in megabytes when `store_type` is `cached_host_mem` or `cached_ssd`|8192|
|amp|enable Automatic Mixed Precision(AMP) training|False|
|loss_scale_policy|loss scale policy for AMP training: `static` or `dynamic`|static|

## Getting started
If you'd like to quickly train a OneFlow DCN model, please follow steps below:
### Installing OneFlow and Dependencies
To install nightly release of OneFlow with CUDA 11.5 support:
```
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu115
```
For more information how to install Oneflow, please refer to [Oneflow Installation Tutorial](
https://github.com/Oneflow-Inc/oneflow#install-oneflow).

Please check `requirements.txt` to install dependencies manually or execute:
```bash
python3 -m pip install -r requirements.txt
```
### Preparing dataset
The Criteo dataset is from [2014-kaggle-display-advertising-challenge-dataset](https://www.kaggle.com/competitions/criteo-display-ad-challenge/overview), considered the original download link is invalid, click [here](https://www.kaggle.com/datasets/mrkmakr/criteo-dataset) to donwload if you would.

Each sample contains:
- Label - Target variable that indicates if an ad was clicked (1) or not (0).
- I1-I13 - A total of 13 columns of integer features (mostly count features).
- C1-C26 - A total of 26 columns of categorical features. The values of these features have been hashed onto 32 bits for anonymization purposes.

We have two methods for data processing

### Data processing 1:
Follow the [FuxiCTR data processing](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/DCN/DCN_criteo_x4_001) method:
1. Use `tools/split_criteo.py` to split train, valid, test csv files
2. Use `tools/make_criteo.py` to make criteo dataset of parquet format, modifying `yourpath/` in  `tools/dataset_config.yaml` and `tools/make_criteo.py` is neeeded.
please check the important arguements below
- `spark_tmp_dir`: change the tmp directory used by pyspark, SSD of 2T or above is recommended
- `spark_driver_memory_gb`: amount of gigabyte memory to use for the driver process, 1024 as default
- `mod_idx`, limited value of index count of each features, `0` or less stands for no limit
- `export_dataset_info`, export `README.md` file in `output_dir` contains subsets count and table size array
3. Please install `pyspark` before running

### Data processing 2:
1. Download the [Criteo Kaggle dataset](https://www.kaggle.com/c/criteo-display-ad-challenge) and then split it using split_criteo_kaggle.py.

2. launch a spark shell using [launch_spark.sh](https://github.com/Oneflow-Inc/models/blob/dev_deepfm_multicol_oneemb/RecommenderSystems/deepfm/tools/launch_spark.sh).

     -   Modify the SPARK_LOCAL_DIRS as needed

         ```shell
         export SPARK_LOCAL_DIRS=/path/to/your/spark/
         ```

     -   Run `bash launch_spark.sh`

3. load [deepfm_parquet.scala](https://github.com/Oneflow-Inc/models/blob/dev_deepfm_multicol_oneemb/RecommenderSystems/deepfm/tools/dcn_parquet.scala) to your spark shell by `:load ddn_parquet.scala`.

4. call the `makeDCNDataset(srcDir: String, dstDir:String)` function to generate the dataset.

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


## Start training by Oneflow
Following command will launch 8 oneflow DCN training and evaluation processes on a node with 8 GPU devices, by specify `data_dir` for data input and `persistent_path` for OneEmbedding persistent store path.

`table_size_array` is close related to sparse features of data input. each sparse field such as `C1` or other `C*` field in criteo dataset corresponds to a embedding table and has its own capacity of unique feature ids, this capacity is also called `number of rows` or `size of embedding table`, the embedding table will be initialized by this value. `table_size_array` holds all sparse fields' `size of embedding table`. `table_size_array` is also used to estimate capacity for OneEmbedding. 

```python
DEVICE_NUM_PER_NODE=1
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
DATA_DIR=your_path/criteo_parquet
PERSISTENT_PATH=your_path/persistent1
MODEL_SAVE_DIR=your_path/model_save_dir

python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    dcn_train_eval.py \
      --data_dir $DATA_DIR \
      --model_save_dir $MODEL_SAVE_DIR \
      --persistent_path $PERSISTENT_PATH \
      --save_initial_model \
      --save_model_after_each_eval \
      --table_size_array "43, 98, 121, 41, 219, 112, 79, 68, 91, 5, 26, 36, 70, 1447, 554, 157461, 117683, 305, 17, 11878, 629, 4, 39504, 5128, 156729, 3175, 27, 11070, 149083, 11, 4542, 1996, 4, 154737, 17, 16, 52989, 81, 40882" \
      --store_type 'cached_host_mem' \
      --dnn_hidden_units "1000, 1000, 1000, 1000, 1000" \
      --crossing_layers 4\
      --net_dropout 0.2 \
      --learning_rate 0.001 \
      --embedding_vec_size 16 \
      --cache_memory_budget_mb 2048 \
      --num_train_samples 36672493 \
      --num_valid_samples 4584062 \
      --num_test_samples 4584062 \
      --train_batch_size 10000 \
      --train_batches 70000 \
      --loss_print_interval 100 \
      --valid_batch_size 10000 \
      --valid_batches 1000 \
      --test_batch_size 10000 \
      --test_batches 1000 \
      --loss_print_interval 100 \
      --size_factor 3

```





