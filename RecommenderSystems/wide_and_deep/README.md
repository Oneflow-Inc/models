# Wide&Deep
[Wide&Deep](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html) is an industrial application model for CTR recommendation that combines Deep Neural Network and Linear model. Its model structure is as follows. Based on this structure, this project uses OneFlow distributed deep learning framework to realize training the modle in graph mode and eager mode respectively on Crioteo data set. 
![1](https://1.bp.blogspot.com/-Dw1mB9am1l8/V3MgtOzp3uI/AAAAAAAABGs/mP-3nZQCjWwdk6qCa5WraSpK8A7rSPj3ACLcB/s1600/image04.png)
## Directory description
```
.
|-- models
    |-- wide_and_deep.py     #Wide&Deep model structure
    |-- data.py         #Read data
|-- utils
    |-- logger.py     #Loger info
|-- config.py                   #Argument configuration
|-- train.py              #Python script for train mode
|-- train_consistent_eager.sh              #Shell script for starting training in eager mode
|-- train_consistent_graph.sh              #Shell script for starting training in graph mode
|-- train_ddp.sh              #Shell script for starting training in ddp mode
|-- __init__.py
└── README.md                   #Documentation
```
## Arguments description
|Argument Name|Argument Explanation|Default Value|
|-----|---|------|
|batch_size|the data batch size in one step training|16384|
|data_dir|the data file directory|/dataset/wdl_ofrecord/ofrecord|
|dataset_format|ofrecord format data or onerec format data|ofrecord|
|deep_dropout_rate|the argument dropout in the deep part|0.5|
|deep_embedding_vec_size|the embedding dim in deep part|16|
|deep_vocab_size|the embedding size in deep part|1603616|
|wide_vocab_size|the embedding size in wide part|1603616|
|hidden_size|number of neurons in every nn layer in the deep part|1024|
|hidden_units_num|number of nn layers in deep part|7|
|learning_rate|argument learning rate|0.001|
|max_iter|maximum number of training batch times|30000|
|model_load_dir|model loading directory||
|model_save_dir|model saving directory||
|num_deep_sparse_fields|number of sparse id features|26|
|num_dense_fields|number of dense features|13|
|loss_print_every_n_iter|print train loss and validate the model after training every number of batche times|1000|
|save_initial_model|save the initial arguments of the modelor not|False|

## Prepare running
### Environment
Running Wide&Deep model requires downloading [OneFlow](https://github.com/Oneflow-Inc/oneflow), [scikit-learn](https://scikit-learn.org/stable/install.html) for caculating mertics, and tool package [numpy](https://numpy.org/)。


### Dataset
[Criteo](https://figshare.com/articles/dataset/Kaggle_Display_Advertising_Challenge_dataset/5732310) dataset is the online advertising dataset published by criteo labs. It contains the function value and click feedback of millions of display advertisements, which can be used as the benchmark for CTR prediction. Each advertisement has the function of describing data. The dataset has 40 attributes. The first attribute is the label, where a value of 1 indicates that the advertisement has been clicked and a value of 0 indicates that the advertisement has not been clicked. This attribute contains 13 integer columns and 26 category columns.

### Prepare ofrecord format data 
Please view [how_to_make_ofrecord_for_wdl](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/ClickThroughRate/WideDeepLearning/how_to_make_ofrecord_for_wdl.md)

## Start training by Oneflow

### Train by using ddp
```
bash train_ddp.sh
```
### Train by graph mode in consistent view 
```
bash train_consistent_graph.sh
```
### Train by eager mode in consistent view 
```
bash train_consistent_eager.sh
```








