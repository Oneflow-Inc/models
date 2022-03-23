# DLRM
[DLRM](https://arxiv.org/pdf/1906.00091.pdf) is a deep learning-based recommendation model that exploits categorical data model for CTR recommendation. Its model structure is as follows. Based on this structure, this project uses OneFlow distributed deep learning framework to realize training the modle in graph mode respectively on Crioteo data set.
![image](https://user-images.githubusercontent.com/63446546/158937131-1a057659-0d49-4bfb-aee2-5568e605fa01.png)

## Directory description
```
.
|-- dlrm_train_eval.py   #OneFlow DLRM training and evaluation scripts with OneEmbedding module. 
|-- requirements.txt     #python package configuration file
└── README.md            #Documentation
```
## Arguments description
|Argument Name|Argument Explanation|Default Value|
|-----|---|------|
|data_dir|the data file directory|*Required Argument*|
|persistent_path|path for OneEmbeddig persistent kv store|*Required Argument*|
|table_size_array|table size array for sparse fields|*Required Argument*|
|store_type|OneEmbeddig persistent kv store type: `device_mem`, `cached_host_mem` or `cached_ssd` |cached_ssd|
|cache_memory_budget_mb|size of cache memory budget on each device in megabytes when `store_type` is `cached_host_mem` or `cached_ssd`|8192|
|embedding_vec_size|embedding vector dimention size|128|
|bottom_mlp|bottom MLPs hidden units number|512,256,128|
|top_mlp|top MLPs hidden units number|1024,1024,512,256|
|disable_interaction_padding|disable interaction output padding or not|False|
|interaction_itself|interaction itself or not|False|
|disable_fusedmlp|disable fused MLP or not|False|
|train_batch_size|training batch size|55296|
|max_iter|number of minibatch training interations|75000|
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
For more information how to install OneFlow, refer to [OneFlow github](
https://github.com/Oneflow-Inc/oneflow#install-oneflow).

Please check `requirements.txt` to install dependencies manually or execute:
```bash
python3 -m pip install -r requirements.txt
```
### Preparing dataset 
- [ ] TODO: scala or pyspark scripts

## Start training by Oneflow
```python
python3 -m oneflow.distributed.launch \
    --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    train_eval.py \
      --data_dir /path/to/dlrm_parquet \
      --persistent_path /path/to/persistent \
      --table_size_array "39884407,39043,17289,7420,20263,3,7120,1543,63,38532952,2953546,403346,10,2208,11938,155,4,976,14,39979772,25641295,39664985,585935,12972,108,36"
```






