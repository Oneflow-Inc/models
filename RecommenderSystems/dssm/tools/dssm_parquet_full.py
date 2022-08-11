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
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.functions import rand, udf, lit, xxhash64, col
from pyspark.sql.types import FloatType


def gen_data_set(args, data):
    seq_max_len = args.seq_len
    negsample = args.negsample

    data.sort_values("timestamp", inplace=True)

    train_set = []
    test_set = []
    for user_id, records in tqdm(data.groupby("user_id")):
        pos_movie_list = records["movie_id"].tolist()
        genres_list = records["genres"].tolist()
        gender_list = records["gender"].tolist()
        age_list = records["age"].tolist()
        occup_list = records["occupation"].tolist()
        zip_list = records["zip"].tolist()

        if negsample > 0:
            item_ids = data["movie_id"].unique()
            item_id_genres_map = dict(zip(data["movie_id"].values, data["genres"].values))
            candidate_set = list(
                set(item_ids) - set(pos_movie_list)
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

    random.shuffle(train_set)
    random.shuffle(test_set)

    train_df = pd.DataFrame(
        train_set,
        columns=[
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
        ],
    )
    test_df = pd.DataFrame(
        test_set,
        columns=[
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
        ],
    )

    print(f"Saving to {args.output_dir}")
    train_file = os.path.join(args.output_dir, "train.csv")
    train_df.to_csv(train_file, index=False)
    test_file = os.path.join(args.output_dir, "test.csv")
    test_df.to_csv(test_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to downloaded and unzipd criteo terabyte datasets: day_0, day_1, ..., day_23",
    )
    parser.add_argument("--output_dir", type=str, required=True)
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

    user_data = pd.read_csv(
        os.path.join(args.input_dir, "users.dat"),
        sep="::",
        names=["user_id", "gender", "age", "occupation", "zip"],
    )
    movie_data = pd.read_csv(
        os.path.join(args.input_dir, "movies.dat"),
        sep="::",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    rating_data = pd.read_csv(
        os.path.join(args.input_dir, "ratings.dat"),
        sep="::",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )

    tmp = pd.merge(rating_data, movie_data, how="left", on="movie_id", validate="many_to_one")
    data = pd.merge(tmp, user_data, how="left", on="user_id", validate="many_to_one")

    sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip", "genres"]

    feature_max_idx = {}
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1

    gen_data_set(args, data)
