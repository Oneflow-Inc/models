# DLRM
[DLRM](https://arxiv.org/pdf/1906.00091.pdf) is a deep learning-based recommendation model that exploits categorical data model for CTR recommendation. Its model structure is as follows. Based on this structure, this project uses OneFlow distributed deep learning framework to realize training the modle in graph mode and eager mode respectively on Crioteo data set.
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
|mlp_type|MLP or FusedMLP|FusedMLP|
|bottom_mlp||512,256,128|
|top_mlp||1024,1024,512,256|
|output_padding|interaction output padding size|1|
|interaction_itself|interaction itself or not||
|model_load_dir|model loading directory||
|model_save_dir|model saving directory|./checkpoint|
|save_initial_model|save initial model parameters or not.||
|save_model_after_each_eval|save model after each eval||
|eval_after_training|do eval after_training||
|data_dir|the data file directory|/dataset/dlrm_parquet|
|eval_batchs|<0: whole val ds, 0: do not val, >0: number of eval batches|-1|
|eval_batch_size||512|
|eval_batch_size_per_proc||None|
|eval_interval||1000|    
|batch_size|the data batch size in one step training|16384|
|batch_size_per_proc||None|
|learning_rate|argument learning rate|1e-3|
|warmup_batches||2750|
|decay_batches||27772|
|decay_start||49315|
|vocab_size||-1|
|embedding_vec_size||128|
|num_dense_fields||13|
|max_iter|maximum number of training batch times|30000|
|loss_print_every_n_iter|print train loss and validate the model after training every number of batche times|100|
|num_sparse_fields||26|
|embedding_type|OneEmbedding or Embedding|OneEmbedding|
|embedding_split_axis|-1: no split|-1|
|column_size_array|column_size_array||
|persistent_path|path for persistent kv store||
|cache_type||device_host|
|cache_memory_budget_mb||16384,16384|
|use_fp16|Run model with amp||
|loss_scale_policy|static or dynamic|static|
|test_name||noname_test|
|data_dir|the data file directory|None|
|learning_rate|argument learning rate|24|
|max_iter|maximum number of training batch times|75000|
|model_load_dir|model loading directory|None|
|model_save_dir|model saving directory|None|
|loss_print_interval|print train loss and validate the model after training every number of batche times|1000|
|save_initial_model|save the initial arguments of the modelor not|False|


- [ ] TODO: other parameters

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






