# Wide&Deep
[Wide&Deep](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html)是一个结合了深度神经网络与线性模型的用于CTR推荐的一个工业应用模型，其模型结构如下，本项目基于该结构，在Crioteo数据集上，使用Oneflow实现了Graph模式与Eager模式的训练。
![1](https://1.bp.blogspot.com/-Dw1mB9am1l8/V3MgtOzp3uI/AAAAAAAABGs/mP-3nZQCjWwdk6qCa5WraSpK8A7rSPj3ACLcB/s1600/image04.png)
## 目录说明
```
.
├── config.py                   #参数配置
├── dataloader_utils.py         #数据读取
├── eager_train.py              #eager模式的训练代码
├── eager_train.sh              #eager模式的训练脚本
├── graph_train.py              #graph模式的训练代码
├── graph_train.sh              #graph模式的训练脚本
├── __init__.py
├── README.md                   #说明文档
├── util.py                     #定义了一些工具方法
└── wide_and_deep_module.py     #Wide&Deep模型定义
```
## 参数说明
|参数名|参数说明|默认值|
|-----|---|------|
|batch_size|批次大小|16384|
|data_dir|数据所在目录|/dataset/wdl_ofrecord/ofrecord|
|dataset_format|ofrecord格式数据或者onerec格式数据|ofrecord|
|deep_dropout_rate|模型deep结构dropout参数|0.5|
|deep_embedding_vec_size|模型deep结构的embedding维度|16|
|deep_vocab_size|模型deep结构的embedding词表大小|1603616|
|wide_vocab_size|模型wide结构的embedding词表大小|1603616|
|hidden_size|模型deep结构nn层神经元个数|1024|
|hidden_units_num|模型deep结构nn层数量|7|
|learning_rate|模型的学习率参数|0.001|
|max_iter|最大训练批次次数|30000|
|model_load_dir|模型加载目录||
|model_save_dir|模型保存目录||
|num_deep_sparse_fields|sparse id特征的个数|26|
|num_dense_fields|dense特征的个数|13|
|print_interval|每隔多少批次打印模型训练loss并进行验证|1000|
|save_initial_model|是否保存模型初始时的参数|False|

## 运行前准备
### 环境
运行Wide&Deep模型前需要安装[OneFlow](https://github.com/Oneflow-Inc/oneflow), [scikit-learn](https://scikit-learn.org/stable/install.html) 用于计算评价指标，以及[numpy](https://numpy.org/)。


### 数据集
[Criteo](https://figshare.com/articles/dataset/Kaggle_Display_Advertising_Challenge_dataset/5732310)数据集是Criteo Labs发布的在线广告数据集。 它包含数百万个展示广告的功能值和点击反馈，该数据可作为点击率(CTR)预测的基准。 每个广告都有描述数据的功能。 数据集具有40个属性，第一个属性是标签，其中值1表示已单击广告，而值0表示未单击广告。 该属性包含13个整数列和26个类别列。

### ofrecord格式的数据准备
可见[how_to_make_ofrecord_for_wdl](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/ClickThroughRate/WideDeepLearning/how_to_make_ofrecord_for_wdl.md)

## 启动Oneflow训练
### Eager模式训练脚本
```
bash eager_train.sh
```
### Graph模式训练脚本
```
bash graph_train.sh
```









