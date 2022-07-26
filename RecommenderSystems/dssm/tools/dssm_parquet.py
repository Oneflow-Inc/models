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

import pandas as pd
from preprocess import gen_data_set, gen_model_input
from sklearn.preprocessing import LabelEncoder

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.functions import rand, udf, lit, xxhash64, col
from pyspark.sql.types import FloatType
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler, VectorAssembler


def make_mmoe_parquet(spark, input_files, output_dir, part_num=None, shuffle=False):
    start = time.time()

    data = pd.read_csvdata = pd.read_csv("./movielens_sample.txt")
    sparse_features = ["movie_id", "user_id",
                       "gender", "age", "occupation", "zip", "genres"]
    SEQ_LEN = 50
    negsample = 10

    feature_max_idx = {}
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1

    user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')

    item_profile = data[["movie_id", "genres"]].drop_duplicates('movie_id')

    user_profile.set_index("user_id", inplace=True)

    user_item_list = data.groupby("user_id")['movie_id'].apply(list)

    train_set, test_set = gen_data_set(data, SEQ_LEN, negsample)

    train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
    test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)

    make_dense = udf(lambda s: float(s), FloatType())
    dense_cols = [make_dense(field).alias(field) for field in dense_features]

    make_label = udf(lambda s: float(s), FloatType())
    label_cols = [make_label(field).alias(field) for field in ["label_income", "label_marital"]]

    sparse_cols = [xxhash64(field, lit(i)).alias(field) for i, field in enumerate(sparse_features)]

    data = data.select(dense_cols + sparse_cols + label_cols)

    # scale dense features
    assemblers = [
        VectorAssembler(inputCols=[col], outputCol=col + "_vec") for col in dense_features
    ]
    scalers = [
        MinMaxScaler(inputCol=col + "_vec", outputCol=col + "_scaled") for col in dense_features
    ]
    pipeline = Pipeline(stages=assemblers + scalers)
    scalerModel = pipeline.fit(data)
    data = scalerModel.transform(data)

    scaled_dense_names = {x + "_scaled": x for x in dense_features}
    vec_to_float = udf(lambda v: float(v[0]), FloatType())
    data = data.select(
        [vec_to_float(c).alias(scaled_dense_names[c]) for c in scaled_dense_names.keys()]
        + sparse_features
        + ["label_income", "label_marital"]
    )

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

    test_csv = os.path.join(args.input_dir, "census-income.test")
    train_csv = os.path.join(args.input_dir, "census-income.data")

    # start spark session
    conf = SparkConf()
    conf.set("spark.driver.memory", f"{args.spark_driver_memory_gb}g")
    conf.set("spark.local.dir", args.spark_tmp_dir)
    spark = SparkSession.builder.config(conf=conf).master("local[*]").getOrCreate()

    # create test dataset
    test_output_dir = os.path.join(args.output_dir, "test")
    test_count = make_mmoe_parquet(spark, test_csv, test_output_dir, part_num=32)

    # create train dataset
    train_output_dir = os.path.join(args.output_dir, "train")
    train_count = make_mmoe_parquet(spark, train_csv, train_output_dir, part_num=64, shuffle=True)

    if args.export_dataset_info:
        df = spark.read.parquet(train_output_dir, test_output_dir)
        table_size_array = [df.select(field).distinct().count() for field in sparse_features]
        print(table_size_array)
        with open(os.path.join(args.output_dir, "README.md"), "w") as f:
            f.write("## number of examples:\n")
            f.write(f"train: {train_count}\n")
            f.write(f"test: {test_count}\n")
            f.write("## table size array\n")
            f.write("table_size_array = [")
            f.write(", ".join([str(i) for i in table_size_array]))
            f.write("]\n")
