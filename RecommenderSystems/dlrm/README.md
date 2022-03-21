# DLRM
[DLRM](https://arxiv.org/pdf/1906.00091.pdf) is a deep learning-based recommendation model that exploits categorical data model for CTR recommendation. Its model structure is as follows. Based on this structure, this project uses OneFlow distributed deep learning framework to realize training the modle in graph mode respectively on Crioteo data set.
![image](https://user-images.githubusercontent.com/63446546/158937131-1a057659-0d49-4bfb-aee2-5568e605fa01.png)

## Directory description
```
.
|-- config.py            #Argument configuration
|-- dataloader.py        #Read data
|-- dlrm.py              #dlrm model structure
|-- train_eval.py        #Shell script for starting training in graph mode
└── README.md            #Documentation
```
## Arguments description
|Argument Name|Argument Explanation|Default Value|
|-----|---|------|
|use_fusedmlp|use fused MLP or not||
|embedding_vec_size||128|
|bottom_mlp||512,256,128|
|top_mlp||1024,1024,512,256|
|disable_interaction_padding|disenable interaction padding or not||
|interaction_itself|interaction itself or not||
|model_load_dir|model loading directory||
|model_save_dir|model saving directory|./checkpoint|
|save_initial_model|save initial model parameters or not.||
|save_model_after_each_eval|save model after each eval||
|not_eval_after_training|do eval after_training||
|data_dir|the data file directory|/dataset/dlrm_parquet|
|eval_batchs|<0: whole val ds, 0: do not val, >0: number of eval batches|-1|
|eval_batch_size||55296|
|eval_batch_size_per_proc||None|
|eval_interval||10000|    
|batch_size|the data batch size in one step training|55296|
|batch_size_per_proc||None|
|learning_rate|argument learning rate|24|
|warmup_batches||2750|
|decay_batches||27772|
|decay_start||49315|
|max_iter|maximum number of training batch times|75000|
|loss_print_every_n_iter|print train loss and validate the model after training every number of batche times|100|
|column_size_array|column_size_array||
|persistent_path|path for persistent kv store||
|store_type|||
|device_memory_budget_mb_per_rank||8192|
|use_fp16|Run model with amp||
|loss_scale_policy|static or dynamic|static|

## Prepare running
### Environment
Running DLRM model requires downloading [OneFlow](https://github.com/Oneflow-Inc/oneflow), [scikit-learn](https://scikit-learn.org/stable/install.html) for caculating mertics, and tool package [numpy](https://numpy.org/)。

- [ ] TODO: petastorm
### Dataset

### Prepare dataset 
- [ ] TODO: scala or pyspark scripts

## Start training by Oneflow
```python
python3 -m oneflow.distributed.launch \
    --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    train_eval.py \
      --persistent_path /path/to/persistent \
      --data_dir /path/to/dlrm_parquet \
      --column_size_array "39884407,39043,17289,7420,20263,3,7120,1543,63,38532952,2953546,403346,10,2208,11938,155,4,976,14,39979772,25641295,39664985,585935,12972,108,36" \
      --use_fulsedmlp
```






