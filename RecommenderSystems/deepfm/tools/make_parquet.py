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
import json
import argparse
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.functions import rand, udf, lit, xxhash64
from pyspark.sql.types import FloatType 


def make_dataframe(spark, input_files):
    sparse_names = [f"C{i}" for i in range(1, 27)]
    dense_names = [f"I{i}" for i in range(1, 14)]
    column_names = ["label"] + dense_names + sparse_names

    make_label = udf(lambda s: float(s), FloatType())
    label_col = make_label("label").alias("label")

    dense_cols = [xxhash64(Ii, lit(i - 1)).alias(Ii) for i, Ii in enumerate(dense_names)]
    sparse_cols = [xxhash64(Ci, lit(i - 1)).alias(Ci) for i, Ci in enumerate(sparse_names, start=len(dense_names))]

    cols = [label_col] + dense_cols + sparse_cols

    df = (
        spark.read.options(delimiter="\t")
        .csv(input_files)
        .toDF(*column_names)
        .select(cols)
    )
    return df


def make_kaggle_parquet(input_dir, output_dir, spark_driver_memory_gb=32):
    train_txt = os.path.join(input_dir, "train.txt")
    assert os.path.isfile(train_txt), f"Can not find train.txt in folder {input_dir}."

    meta = {}

    # start spark session
    conf = SparkConf()
    conf.set("spark.driver.memory", f"{spark_driver_memory_gb}g")
    spark = SparkSession.builder.config(conf=conf).master("local[*]").getOrCreate()

    print("reading raw train.txt ...")
    df = make_dataframe(spark, train_txt)
    meta["field_dtypes"] = df.dtypes 

    train_df, test_df, val_df = df.randomSplit([0.8, 0.1, 0.1])

    print("saving dataset ...")
    train_df.orderBy(rand()).write.mode("overwrite").parquet(os.path.join(output_dir, "train"))
    test_df.write.mode("overwrite").parquet(os.path.join(output_dir, "test"))
    val_df.write.mode("overwrite").parquet(os.path.join(output_dir, "val"))

    print("calculating number of samples ...")
    meta["num_train_samples"] = train_df.count() 
    meta["num_test_samples"] = test_df.count() 
    meta["num_val_samples"] = val_df.count() 

    print("calculating table size array ...")
    col_names = [f"I{i}" for i in range(1, 14)] + [f"C{i}" for i in range(1, 27)] 
    table_size_array = [df.select(c).distinct().count() for c in col_names]
    meta["table_size_array"] = table_size_array

    print("save meta.json")
    meta_json = os.path.join(output_dir, "meta.json")
    with open(meta_json, "w") as fp:
        json.dump(meta, fp)
    print(meta)
    return meta_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/data",
        help="Path to downloaded and unzipd criteo kaggle datasets: train.txt test.txt",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--spark_driver_memory_gb", type=int, default=32)
    args = parser.parse_args()
    make_kaggle_parquet(args.input_dir, args.output_dir, args.spark_driver_memory_gb)

