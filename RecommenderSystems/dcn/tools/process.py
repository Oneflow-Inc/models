import os
import time
import argparse
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.functions import rand, udf, lit, hash
from pyspark.sql.types import IntegerType, LongType
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import sys


from datetime import datetime

import gc
import argparse
import logging
import os
from pathlib import Path
import yaml
import glob
import pandas as pd


import criteo as datasets




def load_config(config_dir, dataset_id):
    params = dict()
    params['dataset_id'] = dataset_id
    dataset_configs = glob.glob(os.path.join(config_dir, 'dataset_config.yaml'))
    for config in dataset_configs:
        with open(config, 'r') as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if dataset_id in config_dict:
                params.update(config_dict[dataset_id])
                break
    return params


def make_accum_feat(ddf , num_list):
    accum_sum = 0
    ddf.iloc[:,0] = ddf.iloc[:,0].astype('int64')
    ddf.iloc[:,1] = ddf.iloc[:,1].astype('int64')
    for i in range(len(num_list)-1):
        accum_sum += num_list[i]
        ddf.iloc[:,i+2] += accum_sum
        ddf.iloc[:,i+2] = ddf.iloc[:,i+2].astype('int64')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./', help='The config directory.')
    parser.add_argument('--dataset_id', type=str, default="criteo_x4_001")
    args = vars(parser.parse_args())

    params = load_config(args['config'],args['dataset_id'])

    print(params)
    
    feature_encoder = datasets.FeatureEncoder(**params)

    if os.path.exists(feature_encoder.pickle_file):
        print("load pickle feature_encoder.pickle_file")
        feature_encoder = feature_encoder.load_pickle(feature_encoder.pickle_file)
    else:
        feature_encoder.fit(**params)
    train_gen, valid_gen = datasets.data_generator(feature_encoder, stage='train', **params)
    test_gen = datasets.data_generator(feature_encoder, stage='test', **params)



 