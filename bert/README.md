Oneflow-Module版BERT实现，参考pytroch版BERT实现:https://github.com/codertimo/BERT-pytorch

### 1.dataset

#### 1.1 示例数据集
示例数据集和词表文件可直接下载：
[data.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/bert/data.zip) 
下载后，解压至本文件夹即可。

#### 1.2 自定义数据集
如果你已经准备好了自定义数据集文件，如：your_corpus.small，则可以通过以下方式生成词表：
放开文件`bert/dataset/vocab.py`中最后一行的注释：`# build()`，然后，运行以下脚本：
`python3 vocab.py  -c /path/to/your_corpus.small  -o /path/to/your_vocab.small`


### 2.依赖项

#### 2.1 framwork
需要手动编译安装此oneflow分支：
https://github.com/Oneflow-Inc/oneflow/tree/dev_add_flow.utils.data

或编译安装[此commit](https://github.com/Oneflow-Inc/oneflow/commit/ec30a681771b0b0389eb6441d6c864f6e9d4ec43)的oneflow

#### 2.2 requirements
`pip3 install -r requirements.txt`

### 3.Uasge

- train: `bash train.sh`
- test: `bash test.sh`


