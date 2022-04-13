"""
Copyright 2020 The OneFlow Authors. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
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
# os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys

import fuxi_data_utils as datasets
from datetime import datetime
from fuxi_features import FeatureMap, FeatureEncoder
import gc
import argparse
import logging
import os
from pathlib import Path
import yaml
import glob
import pandas as pd

def make_frappe_parquet(
    spark, input_files, output_dir, mod_idx=5000, part_num=None, shuffle=False
):

    sparse_names = ["user","item","daytime","weekday","isweekend","homework","cost","weather","country","city"]
    dense_names = []
    column_names = ["label"] + sparse_names + dense_names

    make_label = udf(lambda s: int(s), IntegerType())
    label_col = make_label("label").alias("label")

    make_dense = udf(lambda s: int(1) if s is None else int(s) + 1, IntegerType())
    dense_cols = [make_dense(dense_name).alias(dense_name) for i, dense_name in enumerate(dense_names)]

    make_sparse = udf(lambda s:  int(s) , IntegerType())
    sparse_cols = [make_sparse(sparse_name).alias(sparse_name) for i, sparse_name in enumerate(sparse_names)]


    cols = [label_col] + dense_cols + sparse_cols

    start = time.time()
    df = spark.read.option('header','true').csv(input_files).toDF(*column_names).select(cols)

    if shuffle:
        df = df.orderBy(rand())
    if part_num:
        df = df.repartition(part_num)
    df.write.mode("overwrite").parquet(output_dir)
    num_examples = spark.read.parquet(output_dir).count()
    print(output_dir, num_examples, f"time elapsed: {time.time()-start:0.1f}")
    return num_examples

def load_config(config_dir):
    params = dict()
    dataset_params = load_dataset_config(config_dir, 'frappe_x1_04e961e9')
    params.update(dataset_params)
    return params

def load_dataset_config(config_dir, dataset_id):
    dataset_configs = glob.glob(os.path.join(config_dir, 'dataset_config.yaml'))
    if not dataset_configs:
        dataset_configs = glob.glob(os.path.join(config_dir, 'dataset_config/*.yaml'))
    for config in dataset_configs:
        with open(config, 'r') as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if dataset_id in config_dict:
                return config_dict[dataset_id]
    raise RuntimeError('dataset_id={} is not found in config.'.format(dataset_id))

def datareader():
    print("load csv data ../frappe_temp/train.csv")
    train_data = pd.read_csv('../frappe_temp/train.csv')
    valid_data = pd.read_csv('../frappe_temp/valid.csv')
    test_data = pd.read_csv('../frappe_temp/test.csv')

    train_nums = len(train_data)
    valid_nums = len(valid_data)
    test_nums = len(test_data)

    print("train:", train_nums)
    print("valid:", valid_nums)
    print("test:", test_nums)

    all_data = train_data.append(valid_data.append(test_data)).reset_index(drop=True).copy()
    
    return all_data, train_nums, valid_nums, test_nums





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
    parser.add_argument('--version', type=str, default='pytorch', help='The model version.')
    parser.add_argument('--config', type=str, default='../config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='FM_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    

    parser.add_argument( "--input_accum_dir",type=str,required=True)
    parser.add_argument("--output_parquet_dir", type=str, required=True)
    parser.add_argument("--spark_tmp_dir", type=str, default=None)
    parser.add_argument("--spark_driver_memory_gb", type=int, default=360)
    parser.add_argument("--mod_idx", type=int, default=5000)
    parser.add_argument(
        "--export_dataset_info", action="store_true", help="export dataset infomation or not"
    )
    args = vars(parser.parse_args())
    print(args)

    # args['expid'] = 'DCN_frappe_x1_013_efa58c31'
    args['config'] = './'
    # args['gpu'] = 0

    params = load_config(args['config'])
    # params['gpu'] = args['gpu']
    # params['version'] = args['version']
    # set_logger(params)
    # logging.info(print_to_json(params))
    # seed_everything(seed=params['seed'])

    # preporcess the dataset
    # dataset = params['dataset_id'].split('_')[0].lower()
    # data_dir = os.path.join(params['data_root'], params['dataset_id'])

    # params['']
    print("bbbbb")

    feature_encoder = FeatureEncoder(**params)

    print("Build feature_map and transform h5 data")
    datasets.build_dataset(feature_encoder, **params)

    ### make accum dataset

    all_data, train_nums, valid_nums, test_nums = datareader()
    sparse_features = list(all_data.columns[1:])
    target = ['label']


    # num_list = [957, 4082, 7, 7, 2, 3, 2, 9, 80, 233] 

    num_list = []
    for i in range(all_data.shape[1]):
        if i==0:
            continue
        num_list.append(len(all_data.iloc[:,i].unique()))
        print(len(all_data.iloc[:,i].unique()))


    make_accum_feat(all_data)

    train_ddf = all_data.iloc[:train_nums,:]
    valid_ddf = all_data.iloc[train_nums:train_nums+valid_nums,:]
    test_ddf = all_data.iloc[train_nums+valid_nums:train_nums+valid_nums+test_nums,:]

    train_ddf.to_csv('../frappe_temp_accum/train.csv',index=False)
    valid_ddf.to_csv('../frappe_temp_accum/valid.csv',index=False)
    test_ddf.to_csv('../frappe_temp_accum/test.csv',index=False)



    ### make parquet frappe ###
    # frappe dataset path
    train_csv = os.path.join(args['input_accum_dir'],"train.csv")
    valid_csv = os.path.join(args['input_accum_dir'],"valid.csv")
    test_csv = os.path.join(args['input_accum_dir'],"test.csv")

    # start spark session
    conf = SparkConf()
    conf.set("spark.driver.memory", f"{360}g")
    conf.set("spark.local.dir", None)
    spark = SparkSession.builder.config(conf=conf).master("local[*]").getOrCreate()


    # create test dataset
    test_output_dir = os.path.join(args['output_parquet_dir'], "test")
    test_count = make_frappe_parquet(
        spark, test_csv, test_output_dir, part_num=256, mod_idx=args['mod_idx']
    )

    # create validation dataset
    val_output_dir = os.path.join(args['output_parquet_dir'], "val")
    val_count = make_frappe_parquet(
        spark, valid_csv, val_output_dir, part_num=256, mod_idx=args['mod_idx']
    )

    # create train dataset
    train_output_dir = os.path.join(args['output_parquet_dir'], "train")
    train_count = make_frappe_parquet(
        spark, train_csv, train_output_dir, part_num=1024, shuffle=True, mod_idx=args['mod_idx']
    )

    if args['export_dataset_info']:
        df = spark.read.parquet(train_output_dir, test_output_dir, val_output_dir)
        sparse_names = ["user","item","daytime","weekday","isweekend","homework","cost","weather","country","city"]
        table_size_array = [df.select(sparse_name).distinct().count() for sparse_name in sparse_names]
        print(table_size_array)
        with open(os.path.join(args['output_parquet_dir'], "README.md"), "w") as f:
            f.write("## number of examples:\n")
            f.write(f"train: {train_count}\n")
            f.write(f"test: {test_count}\n")
            f.write(f"val: {val_count}\n\n")
            f.write("## table size array\n")
            f.write("table_size_array = [")
            f.write(", ".join([str(i) for i in table_size_array]))
            f.write("]\n")