# Deep&Cross
 [Deep & Cross Network](https://dl.acm.org/doi/10.1145/3124749.3124754) (DCN) can not only keep the advantages of DNN model, but also learn specific bounded feature crossover more effectively. In particular, DCN can explicitly learn cross features for each layer without the need for manual feature engineering, and the increased algorithm complexity is almost negligible compared with DNN model.


## Directory description
```
.
|-- tools
  |-- dataset_config.yaml   # dataset config file
  |-- split_criteo.py      # split Criteo file
  |-- make_criteo_parquet.py # make Criteo parquet data from csv data
|-- dcn_train_eval.py       # OneFlow DCN training and evaluation scripts with OneEmbedding module
|-- train.sh                # command to train DCN
|-- requirements.txt         # python package configuration file
└── README.md                # Documentation
```



