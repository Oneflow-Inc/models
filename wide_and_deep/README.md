# Wide&Deep
[Wide&Deep](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html) is an industrial application model for CTR recommendation that combines Deep Neural Network and Linear model. Its model structure is as follows. Based on this structure, this project uses OneFlow distributed deep learning framework to realize training the modle in Graph pattern and Eagle pattern respectively on Crioteo data set. 
![1](https://1.bp.blogspot.com/-Dw1mB9am1l8/V3MgtOzp3uI/AAAAAAAABGs/mP-3nZQCjWwdk6qCa5WraSpK8A7rSPj3ACLcB/s1600/image04.png)
## Directory description
```
.
├── config.py                   #Parameter configuration
├── dataloader_utils.py         #Read data
├── eager_train.py              #Python code for Eager pattern
├── eager_train.sh              #Shell code for starting training in Eager pattern
├── graph_train.py              #Python code for Graph pattern
├── graph_train.sh              #Shell code for starting training in Graph pattern
├── __init__.py
├── README.md                   #Documentation
├── util.py                     #Some utility methods are defined
└── wide_and_deep_module.py     #Wide&Deep model structure
```
## Parameters description
|Parameter Name|Parameter Explanation|Default Value|
|-----|---|------|
|batch_size|the data batch size in one step training|16384|
|data_dir|the data file directory|/dataset/wdl_ofrecord/ofrecord|
|dataset_format|ofrecord format data or onerec format data|ofrecord|
|deep_dropout_rate|the parameter dropout in the deep part|0.5|
|deep_embedding_vec_size|the embedding dim in deep part|16|
|deep_vocab_size|the embedding size in deep part|1603616|
|wide_vocab_size|the embedding size in wide part|1603616|
|hidden_size|number of neurons in every nn layer in the deep part|1024|
|hidden_units_num|number of nn layers in deep part|7|
|learning_rate|parameter learning rate|0.001|
|max_iter|maximum number of training batch times|30000|
|model_load_dir|model loading directory||
|model_save_dir|model saving directory||
|num_deep_sparse_fields|number of sparse id features|26|
|num_dense_fields|number of dense features|13|
|print_interval|print train loss and validate the model after training every number of batche times|1000|
|save_initial_model|save the initial parameters of the modelor not|False|

## Prepare running
### Environment
Running Wide&Deep model requires downloading [OneFlow](https://github.com/Oneflow-Inc/oneflow), [scikit-learn](https://scikit-learn.org/stable/install.html) for caculating mertics, and tool package [numpy](https://numpy.org/)。


### Dataset
[Criteo](https://figshare.com/articles/dataset/Kaggle_Display_Advertising_Challenge_dataset/5732310) dataset is the online advertising dataset published by criteo labs. It contains the function value and click feedback of millions of display advertisements, which can be used as the benchmark for CTR prediction. Each advertisement has the function of describing data. The dataset has 40 attributes. The first attribute is the label, where a value of 1 indicates that the advertisement has been clicked and a value of 0 indicates that the advertisement has not been clicked. This attribute contains 13 integer columns and 26 category columns.

### Prepare ofrecord format data 
Please view [how_to_make_ofrecord_for_wdl](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/ClickThroughRate/WideDeepLearning/how_to_make_ofrecord_for_wdl.md)

## Start training by Oneflow
### Training in Eager pattern
```
bash eager_train.sh
```
### Training in Graph pattern
```
bash graph_train.sh
```









