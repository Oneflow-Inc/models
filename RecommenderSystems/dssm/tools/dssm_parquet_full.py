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

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.functions import rand, udf, lit, xxhash64, col
from pyspark.sql.types import FloatType, ArrayType, LongType


def gen_data_set(
    args,
    data,
    user_sparse,
    item_sparse,
    spark,
    train_part_num=None,
    test_part_num=None,
    train_shuffle=True,
    test_shuffle=True,
):
    seq_max_len = args.seq_len
    negsample = args.negsample

    data = data.sort(data.timestamp.asc())

    train_set = []
    test_set = []
    for row in data.select("user_id").distinct().collect():
        user_id = row.user_id
        filtered_data = data.where(data.user_id == user_id)
        pos_movie_list = [tmp.movie_id for tmp in filtered_data.select("movie_id").collect()]
        genres_list = [tmp.genres for tmp in filtered_data.select("genres").collect()]
        gender_list = [tmp.gender for tmp in filtered_data.select("gender").collect()]
        age_list = [tmp.age for tmp in filtered_data.select("age").collect()]
        occup_list = [tmp.occupation for tmp in filtered_data.select("occupation").collect()]
        zip_list = [tmp.zip for tmp in filtered_data.select("zip").collect()]

        if negsample > 0:
            uni_movie_ids = [tmp.movie_id for tmp in data.select("movie_id").distinct().collect()]
            all_movie_ids = [tmp.movie_id for tmp in data.select("movie_id").collect()]
            all_genres_ids = [tmp.genres for tmp in data.select("genres").collect()]
            item_id_genres_map = dict(zip(all_movie_ids, all_genres_ids))
            candidate_set = list(
                set(uni_movie_ids) - set(pos_movie_list)
            )  # find those not in the user's selection
            neg_movie_list = np.random.choice(
                candidate_set, size=len(pos_movie_list) * negsample, replace=True
            )

        for i in range(1, len(pos_movie_list)):
            pos_movie_hist = pos_movie_list[:i]
            genres_hist = genres_list[:i]
            seq_len = min(i, seq_max_len)
            if i != len(pos_movie_list) - 1:
                train_set.append(
                    (
                        1,
                        user_id,
                        seq_len,
                        pos_movie_list[i],
                        pos_movie_hist[::-1][:seq_len] + [0] * (seq_max_len - seq_len),
                        genres_list[i],
                        genres_hist[::-1][:seq_len] + [0] * (seq_max_len - seq_len),
                        gender_list[i],
                        age_list[i],
                        occup_list[i],
                        zip_list[i],
                    )
                )
                for neg_i in range(negsample):
                    train_set.append(
                        (
                            0,
                            user_id,
                            seq_len,
                            neg_movie_list[i * negsample + neg_i],
                            pos_movie_hist[::-1][:seq_len] + [0] * (seq_max_len - seq_len),
                            item_id_genres_map[neg_movie_list[i * negsample + neg_i]],
                            genres_hist[::-1][:seq_len] + [0] * (seq_max_len - seq_len),
                            gender_list[i],
                            age_list[i],
                            occup_list[i],
                            zip_list[i],
                        )
                    )
            else:
                # one test sample for each user
                test_set.append(
                    (
                        1,
                        user_id,
                        seq_len,
                        pos_movie_list[i],
                        pos_movie_hist[::-1][:seq_len] + [0] * (seq_max_len - seq_len),
                        genres_list[i],
                        genres_hist[::-1][:seq_len] + [0] * (seq_max_len - seq_len),
                        gender_list[i],
                        age_list[i],
                        occup_list[i],
                        zip_list[i],
                    )
                )

    columns = [
        "label",
        "user_id",
        "seq_len",
        "movie_id",
        "movie_hist",
        "genres",
        "genres_hist",
        "gender",
        "age",
        "occupation",
        "zip",
    ]

    make_sparse = udf(lambda s: int(s), LongType())
    user_sparse_cols = [make_sparse(field).alias(field) for field in user_sparse]
    item_sparse_cols = [make_sparse(field).alias(field) for field in item_sparse]

    make_seq = udf(lambda s: list(s), ArrayType(LongType()))
    user_seq_cols = [make_seq(field).alias(field) for field in ["movie_hist", "genres_hist"]]

    make_dense = udf(lambda s: float(s), FloatType())
    dense_cols = [make_dense(field).alias(field) for field in ["label", "seq_len"]]

    train_df = spark.createDataFrame(data=train_set, schema=columns).select(
        dense_cols + user_sparse_cols + user_seq_cols + item_sparse_cols
    )
    test_df = spark.createDataFrame(data=test_set, schema=columns).select(
        dense_cols + user_sparse_cols + user_seq_cols + item_sparse_cols
    )
    train_df.printSchema()
    test_df.printSchema()
    if train_shuffle:
        train_df = train_df.orderBy(rand())
    if test_shuffle:
        test_df = test_df.orderBy(rand())
    if train_part_num:
        train_df = train_df.repartition(train_part_num)
    if test_part_num:
        test_df = test_df.repartition(test_part_num)

    train_df.write.mode("overwrite").parquet(os.path.join(args.output_dir, "train"))
    test_df.write.mode("overwrite").parquet(os.path.join(args.output_dir, "test"))
    print(output_dir, f"time elapsed: {time.time()-start:0.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to downloaded and unziped movielens ml-1m datasets",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--merged_dataset_dir", type=str, default=None, help="path to merged ml-1m dataset ml-1m.csv")
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument(
        "--negsample", type=int, default=10, help="num_pos_sample : num_neg_sample = 1 : negsample"
    )
    parser.add_argument("--spark_tmp_dir", type=str, default=None)
    parser.add_argument("--spark_driver_memory_gb", type=int, default=360)
    parser.add_argument(
        "--export_dataset_info", action="store_true", help="export dataset infomation or not"
    )
    args = parser.parse_args()

    # start spark session
    conf = SparkConf()
    conf.set("spark.driver.memory", f"{args.spark_driver_memory_gb}g")
    conf.set("spark.local.dir", args.spark_tmp_dir)
    spark = SparkSession.builder.config(conf=conf).master("local[*]").getOrCreate()


    if args.merged_dataset_dir:
        data = (
            spark.read.format("csv")
            .option("header", "true")
            .load(os.path.join(args.merged_dataset_dir, "ml-1m.csv"))
        )
    else:
        user_col_name = ["user_id", "gender", "age", "occupation", "zip"]
        user_data = (
            spark.read.format("csv")
            .option("header", "false")
            .option("delimiter", "::")
            .load(os.path.join(args.input_dir, "users.dat"))
            .toDF(*user_col_name)
        )

        movie_col_name = ["movie_id", "title", "genres"]
        movie_data = (
            spark.read.format("csv")
            .option("header", "false")
            .option("delimiter", "::")
            .load(os.path.join(args.input_dir, "movies.dat"))
            .toDF(*movie_col_name)
        )

        rating_col_name = ["user_id", "movie_id", "rating", "timestamp"]
        rating_data = (
            spark.read.format("csv")
            .option("header", "false")
            .option("delimiter", "::")
            .load(os.path.join(args.input_dir, "ratings.dat"))
            .toDF(*rating_col_name)
        )

        tmp = (
            rating_data.alias("r")
            .join(movie_data.alias("m"), col("r.movie_id") == col("m.movie_id"), "left")
            .drop(col("m.movie_id"))
        )
        data = (
            tmp.alias("t")
            .join(user_data.alias("u"), col("t.user_id") == col("u.user_id"), "left")
            .drop(col("u.user_id"))
        )

    user_sparse = ["user_id", "gender", "age", "occupation", "zip"]
    item_sparse = ["movie_id", "genres"]
    sparse_cols = [
        xxhash64(field, lit(i)).alias(field) for i, field in enumerate(user_sparse + item_sparse)
    ]

    make_dense = udf(lambda s: float(s), FloatType())
    dense_cols = [make_dense(field).alias(field) for field in ["rating", "timestamp"]]

    data = data.select(sparse_cols + dense_cols)
    print("Merged dataset schema:")
    data.printSchema()

    if not args.merged_dataset_dir:
        print("Saving merged dataset to {args.output_dit}/ml-1m.csv")
        data.toPandas().to_csv(os.path.join(args.output_dir, "ml-1m.csv"), index=False)

    gen_data_set(args, data, user_sparse, item_sparse, spark, train_part_num=128, test_part_num=32)
