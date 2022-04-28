import h5py
import os
import io
import json
from collections import Counter, OrderedDict
import numpy as np
import time
import argparse
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.functions import rand, udf, lit, xxhash64
from pyspark.sql.types import IntegerType, LongType

def load_hdf5(data_path, key='data'):
    with h5py.File(data_path, 'r') as hf:
            data_array = hf[key][:]
    return data_array


def make_fuxih5_to_parquet(spark, feature_map, data_path, output_dir, part_num=None, shuffle=False):
    print("start loading h5 data!")
    data_array = load_hdf5(data_path) # cols: 39 features + 1 label
    X = data_array[:, :-1]
    label = data_array[:, -1].reshape(-1, 1)

    print("start transforming h5 data!")
    total_prev_vocab = 0
    for key in feature_map['feature_specs'].keys():
        X[:, feature_map['feature_specs'][key]['index']] += total_prev_vocab
        total_prev_vocab += float(feature_map['feature_specs'][key]['vocab_size'])
    
    print("start creating dataframe!")
    data_array = np.concatenate((label, X), axis=1)
    data_pandas = pd.DataFrame(data_array)
    print("pd done!")

    sparse_names = [f"C{i}" for i in range(1, 27)]
    dense_names = [f"I{i}" for i in range(1, 14)]
    columns = ["Label"] + dense_names + sparse_names

    start = time.time()
    df = spark.createDataFrame(data_pandas, schema=columns)
    print("spark done!")

    if shuffle:
        df = df.orderBy(rand())
    if part_num:
        df = df.repartition(part_num)

    print("start writing parquet data!")
    df.write.mode("overwrite").parquet(output_dir)
    num_examples = spark.read.parquet(output_dir).count()
    print(output_dir, num_examples, f"time elapsed: {time.time()-start:0.1f}")
    return num_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to downloaded and unzipd criteo terabyte datasets: day_0, day_1, ..., day_23",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--spark_tmp_dir", type=str, default=None)
    parser.add_argument("--spark_driver_memory_gb", type=int, default=360)
    parser.add_argument(
        "--export_dataset_info", action="store_true", help="export dataset infomation or not"
    )
    args = parser.parse_args()

    feature_map_json = os.path.join(args.input_dir, "feature_map.json")
    test_h5 = os.path.join(args.input_dir, "test.h5")
    val_h5 = os.path.join(args.input_dir, "valid.h5")
    train_h5 = os.path.join(args.input_dir, "train.h5")

    print("Start Loading feature map!")
    with io.open(feature_map_json, "r", encoding="utf-8") as fd:
        feature_map = json.load(fd, object_pairs_hook=OrderedDict)
    print("Loading feature map done!")

    # start spark session
    conf = SparkConf()
    conf.set("spark.driver.memory", f"{args.spark_driver_memory_gb}g")
    conf.set("spark.local.dir", args.spark_tmp_dir)
    conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark = SparkSession.builder.config(conf=conf).master("local[*]").getOrCreate()

    print("start making test parquet data!")
    test_output_dir = os.path.join(args.output_dir, "test")
    test_count = make_fuxih5_to_parquet(
        spark, 
        feature_map, 
        test_h5, 
        test_output_dir, 
        part_num=256, 
        shuffle=False
    )

    print("start making val parquet data!")
    val_output_dir = os.path.join(args.output_dir, "val")
    val_count = make_fuxih5_to_parquet(
        spark, 
        feature_map, 
        val_h5, 
        val_output_dir, 
        part_num=256, 
        shuffle=False
    )

    print("start making train parquet data!")
    train_output_dir = os.path.join(args.output_dir, "train")
    train_count = make_fuxih5_to_parquet(
        spark, 
        feature_map,
        train_h5, 
        train_output_dir, 
        part_num=2048, 
        shuffle=True
    )

    if args.export_dataset_info:
        df = spark.read.parquet(train_output_dir, test_output_dir, val_output_dir)
        table_size_array = [df.select(f"I{i}").distinct().count() for i in range(1, 14)] + [df.select(f"C{i}").distinct().count() for i in range(1, 27)]
        print(table_size_array)
        with open(os.path.join(args.output_dir, "README.md"), "w") as f:
            f.write("## number of examples:\n")
            f.write(f"train: {train_count}\n")
            f.write(f"test: {test_count}\n")
            f.write(f"val: {val_count}\n\n")
            f.write("## table size array\n")
            f.write("table_size_array = [")
            f.write(", ".join([str(i) for i in table_size_array]))
            f.write("]\n")

    