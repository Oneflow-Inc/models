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
from pyspark.sql.functions import rand, udf, lit, xxhash64
from pyspark.sql.types import IntegerType, LongType


def make_dlrm_parquet(
    spark, input_files, output_dir, mod_idx=40000000, part_num=None, shuffle=False
):
    sparse_names = [f"C{i}" for i in range(1, 27)]
    dense_names = [f"I{i}" for i in range(1, 14)]
    column_names = ["label"] + dense_names + sparse_names

    make_label = udf(lambda s: int(s), IntegerType())
    label_col = make_label("label").alias("label")

    make_dense = udf(lambda s: int(1) if s is None else int(s) + 1, IntegerType())
    dense_cols = [make_dense(Ii).alias(Ii) for i, Ii in enumerate(dense_names)]

    if mod_idx <= 0:
        sparse_cols = [xxhash64(Ci, lit(i - 1)).alias(Ci) for i, Ci in enumerate(sparse_names)]
    else:
        make_sparse = udf(
            lambda s, i: mod_idx * i if s is None else int(s, 16) % mod_idx + mod_idx * i,
            LongType(),
        )
        sparse_cols = [make_sparse(Ci, lit(i - 1)).alias(Ci) for i, Ci in enumerate(sparse_names)]

    cols = [label_col] + dense_cols + sparse_cols

    start = time.time()
    df = spark.read.options(delimiter="\t").csv(input_files).toDF(*column_names).select(cols)
    if shuffle:
        df = df.orderBy(rand())
    if part_num:
        df = df.repartition(part_num)
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
    parser.add_argument("--mod_idx", type=int, default=40000000)
    parser.add_argument(
        "--export_dataset_info", action="store_true", help="export dataset infomation or not"
    )
    args = parser.parse_args()

    # split day_23() to test.csv and val.csv
    # 178274637 89137319 89137318
    num_test_examples = 89137319
    day_23 = os.path.join(args.input_dir, "day_23")
    test_csv = os.path.join(args.output_dir, "test.csv")
    val_csv = os.path.join(args.output_dir, "val.csv")
    if not os.path.isfile(test_csv):
        os.system(f"head -n {num_test_examples} {day_23} > {test_csv}")
    if not os.path.isfile(val_csv):
        os.system(f"tail -n +{num_test_examples + 1} {day_23} > {val_csv}")

    # start spark session
    conf = SparkConf()
    conf.set("spark.driver.memory", f"{args.spark_driver_memory_gb}g")
    conf.set("spark.local.dir", args.spark_tmp_dir)
    spark = SparkSession.builder.config(conf=conf).master("local[*]").getOrCreate()

    # create test dataset
    test_output_dir = os.path.join(args.output_dir, "test")
    test_count = make_dlrm_parquet(
        spark, test_csv, test_output_dir, part_num=256, mod_idx=args.mod_idx
    )

    # create validation dataset
    val_output_dir = os.path.join(args.output_dir, "val")
    val_count = make_dlrm_parquet(
        spark, val_csv, val_output_dir, part_num=256, mod_idx=args.mod_idx
    )

    # create train dataset
    train_files = [os.path.join(args.input_dir, f"day_{i}") for i in range(0, 23)]
    train_output_dir = os.path.join(args.output_dir, "train")
    train_count = make_dlrm_parquet(
        spark, train_files, train_output_dir, part_num=1024, shuffle=True, mod_idx=args.mod_idx
    )

    if args.export_dataset_info:
        df = spark.read.parquet(train_output_dir, test_output_dir, val_output_dir)
        table_size_array = [df.select(f"C{i}").distinct().count() for i in range(1, 27)]
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
