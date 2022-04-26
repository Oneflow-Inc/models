# Deep&Cross
 [Deep & Cross Network](https://dl.acm.org/doi/10.1145/3124749.3124754) (DCN) can not only keep the advantages of DNN model, but also learn specific bounded feature crossover more effectively. In particular, DCN can explicitly learn cross features for each layer without the need for manual feature engineering, and the increased algorithm complexity is almost negligible compared with DNN model.


## Directory description
```
.
|-- tools
  |-- criteo.py             # fuxi data preprofile 
  |-- csv_2_h5.py           # fuxi file used to transform data from csv to h5
  |-- dataset_config.yaml   # dataset config file
  |-- fuxi_features.py      # fuxi file
  |--
  |--criteo1t_parquet.py    # Read Criteo1T data and export it as parquet data format
|-- dlrm_train_eval.py       # OneFlow DLRM training and evaluation scripts with OneEmbedding module
|-- requirements.txt         # python package configuration file
└── README.md                # Documentation
```



