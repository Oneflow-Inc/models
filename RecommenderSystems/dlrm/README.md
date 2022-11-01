# DLRM
[DLRM](https://arxiv.org/pdf/1906.00091.pdf) is a deep learning-based recommendation model that exploits categorical data for click-through rate (CTR) prediction and rankings. Its model structure is as follows. Based on this structure, this project uses OneFlow distributed deep learning framework to realize training the model in graph mode on the Criteo data set.
![image](https://user-images.githubusercontent.com/16327185/182550409-f6923d81-d50b-495e-a178-5e2066a1ead3.png)

## Directory description
```
.
|-- tools
  |-- criteo1t_parquet.py    # Read Criteo1T data and export it as parquet data format
|-- dlrm_train_eval.py       # OneFlow DLRM training and evaluation scripts with OneEmbedding module
|-- requirements.txt         # python package configuration file
└── README.md                # Documentation
```

## Arguments description
|Argument Name|Argument Explanation|Default Value|
|-----|---|------|
|data_dir|the data file directory|*Required Argument*|
|persistent_path|path for OneEmbedding persistent kv store|*Required Argument*|
|table_size_array|table size array for sparse fields|*Required Argument*|
|store_type|OneEmbeddig persistent kv store type: `device_mem`, `cached_host_mem` or `cached_ssd` |cached_host_mem|
|cache_memory_budget_mb|size of cache memory budget on each device in megabytes when `store_type` is `cached_host_mem` or `cached_ssd`|8192|
|embedding_vec_size|embedding vector dimention size|128|
|bottom_mlp|bottom MLPs hidden units number|512,256,128|
|top_mlp|top MLPs hidden units number|1024,1024,512,256|
|disable_interaction_padding|disable interaction output padding or not|False|
|interaction_itself|interaction itself or not|False|
|disable_fusedmlp|disable fused MLP or not|False|
|train_batch_size|training batch size|55296|
|train_batches|number of minibatch training interations|75000|
|learning_rate|basic learning rate for training|24|
|warmup_batches|learning rate warmup batches|2750|
|decay_start|learning rate decay start iteration|49315|
|decay_batches|number of learning rate decay iterations|27772|
|loss_print_interval|training loss print interval|1000|
|eval_interval|evaluation interval|10000|
|eval_batches|number of evaluation batches|1612|
|eval_batch_size|evaluation batch size|55296|
|model_load_dir|model loading directory|None|
|model_save_dir|model saving directory|None|
|save_model_after_each_eval|save model or not after each evaluation|False|
|save_initial_model|save initial model parameters or not|False|
|amp|enable Automatic Mixed Precision(AMP) training|False|
|loss_scale_policy|loss scale policy for AMP training: `static` or `dynamic`|static|

## Getting Started
If you'd like to quickly train a OneFlow DLRM model, please follow steps below:
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
[Terabyte Click Logs dataset of CriteoLabs (Criteo1t)](https://labs.criteo.com/2013/12/download-terabyte-click-logs/) contains feature values and click feedback for millions of display ads. Criteo1t contains 24 files, each one corresponding to one day of data.

Each sample contains:
- 1 label, 0 if the ad wasn't clicked and 1 if the ad was clicked
- 13 dense features taking integer values, some values are `-1`
- 26 categorical features, some features may have missing values

In our data preprocess, the label is mapped to integer, literal `1` is added to dense features, there are two options for categorical features:
1. The index count of each features is limited to `mod_idx`(40 million as default), and offset `mod_idx * i` is added to the limited value to make sure each column has different ids, `i` stands for column id.
2. The original 32 bits hashed value is hashed onto 64 bits alone with column id `i` to make sure each column has different ids.

Please find `tools/criteo1t_parquet.py` for more information. Besides `input_dir` and `output_dir`, there are a few more arguments to run `tools/criteo1t_parquet.py`:
- `spark_tmp_dir`: change the tmp directory used by pyspark, SSD of 2T or above is recommended
- `spark_driver_memory_gb`: amount of gigabyte memory to use for the driver process, 360 as default
- `mod_idx`, limited value of index count of each features, `0` or less stands for no limit
- `export_dataset_info`, export `README.md` file in `output_dir` contains subsets count and table size array

Please install `pyspark` before running.

```bash
python tools/criteo1t_parquet.py \
    --input_dir=/path/to/criteo1t/day0-day23 \
    --output_dir=/path/to/dlrm_parquet \
    --spark_tmp_dir=/spark_tmp_dir \
    --export_dataset_info
```

Note 1: The `day_23` file will be splited to two `test.csv` and `val.csv` files in folder `output_dir`.

Note 2: number of examples
- train: 4195197692
- test: 89137319
- val: 89137318

### Make dataset in spark shell
In `Preparing dataset` we introduce pyspark to make dataset for DLRM, the whole process consumes a lot of host memory(>300G) and disk space. In this chapter, we introdce another method, in this method training data is shuffled day by day instead of globally. In this way, host memory is limited to 64G or less. The generated dataset is also qulified, test AUC is 0.8030 around.
In Spark Shell, load `criteo1t_parquet_day_by_day.scala` and make DLRM parquet dataset.
```
scala> :load /path/to/models/RecommenderSystems/dlrm/tools/criteo1t_parquet_day_by_day.scala
scala> makeDlrmDataset("/path/to/criteo1t/day0-day23", "/path/to/dlrm_parquet", "/path/to/tmp_spark")
```

In `/path/to/dlrm_parquet`, move all `parquet` files in folder `shuffled_day_parts` to `train` folder.
```bash
$ mkdir train
$ mv ./shuffled_day_parts/day_part_*/*.parquet train/.
```
Note: in `criteo1t_parquet_day_by_day.scala`, date type of categorical columns C1-C26 is `int32`.

## Start training by Oneflow
Following command will launch 8 oneflow dlrm training and evaluation processes on a node with 8 GPU devices, by specify `data_dir` for data input and `persistent_path` for OneEmbedding persistent store path.

`table_size_array` is close related to sparse features of data input. each sparse field such as `C1` or other `C*` field in criteo dataset corresponds to a embedding table and has its own capacity of unique feature ids, this capacity is also called `number of rows` or `size of embedding table`, the embedding table will be initialized by this value. `table_size_array` holds all sparse fields' `size of embedding table`. `table_size_array` is also used to estimate capacity for OneEmbedding. 

```python
python3 -m oneflow.distributed.launch \
    --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    dlrm_train_eval.py \
      --data_dir /path/to/dlrm_parquet \
      --persistent_path /path/to/persistent
```
