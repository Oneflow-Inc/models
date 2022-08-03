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

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.functions import rand, udf, lit, xxhash64, col
from pyspark.sql.types import FloatType


column_names = ["label", "user_id", "item_id", "tag_id"]
sparse_names = ["user_id"] + ["item_id", "tag_id"]


def make_mmoe_parquet(spark, input_files, output_dir, part_num=None, shuffle=False):
    start = time.time()

    data = spark.read.format("csv").option("header", "True").load(input_files).toDF(*column_names)

    make_label = udf(lambda s: float(s), FloatType())
    label_cols = [make_label("label").alias("label")]

    sparse_cols = [xxhash64(field, lit(i)).alias(field) for i, field in enumerate(sparse_names)]

    data = data.select(label_cols + sparse_cols)

    if shuffle:
        data = data.orderBy(rand())
    if part_num:
        data = data.repartition(part_num)

    data.write.mode("overwrite").parquet(output_dir)
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

    test_csv = os.path.join(args.input_dir, "test.csv")
    valid_csv = os.path.join(args.input_dir, "valid.csv")
    train_csv = os.path.join(args.input_dir, "train.csv")

    # start spark session
    conf = SparkConf()
    conf.set("spark.driver.memory", f"{args.spark_driver_memory_gb}g")
    conf.set("spark.local.dir", args.spark_tmp_dir)
    spark = SparkSession.builder.config(conf=conf).master("local[*]").getOrCreate()

    # create test dataset
    test_output_dir = os.path.join(args.output_dir, "test")
    test_count = make_mmoe_parquet(spark, test_csv, test_output_dir, part_num=32)

    # create valid dataset
    valid_output_dir = os.path.join(args.output_dir, "val")
    valid_count = make_mmoe_parquet(spark, valid_csv, valid_output_dir, part_num=32)

    # create train dataset
    train_output_dir = os.path.join(args.output_dir, "train")
    train_count = make_mmoe_parquet(spark, train_csv, train_output_dir, part_num=64, shuffle=True)

    if args.export_dataset_info:
        df = spark.read.parquet(train_output_dir, valid_output_dir, test_output_dir)
        table_size_array = [df.select(field).distinct().count() for field in sparse_names]
        print(table_size_array)
        with open(os.path.join(args.output_dir, "README.md"), "w") as f:
            f.write("## number of examples:\n")
            f.write(f"train: {train_count}\n")
            f.write(f"valid: {valid_count}\n")
            f.write(f"test: {test_count}\n")
            f.write("## table size array\n")
            f.write("table_size_array = [")
            f.write(", ".join([str(i) for i in table_size_array]))
            f.write("]\n")
