# DLRM
[Wide&Deep](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html) is an industrial application model for CTR recommendation that combines Deep Neural Network and Linear model. Its model structure is as follows. Based on this structure, this project uses OneFlow distributed deep learning framework to realize training the modle in graph mode and eager mode respectively on Crioteo data set. 
![1](https://1.bp.blogspot.com/-Dw1mB9am1l8/V3MgtOzp3uI/AAAAAAAABGs/mP-3nZQCjWwdk6qCa5WraSpK8A7rSPj3ACLcB/s1600/image04.png)
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
|batch_size|the data batch size in one step training|16384|
|data_dir|the data file directory|None|
|learning_rate|argument learning rate|24|
|max_iter|maximum number of training batch times|75000|
|model_load_dir|model loading directory|None|
|model_save_dir|model saving directory|None|
|loss_print_every_n_iter|print train loss and validate the model after training every number of batche times|1000|
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
      --column_size_array "39884407,39043,17289,7420,20263,3,7120,1543,63,38532952,2953546,403346,10,2208,11938,155,4,976,14,39979772,25641295,39664985,585935,12972,108,36"\
      --use_fulsedmlp
```






