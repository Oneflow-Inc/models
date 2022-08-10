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

import random
# from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm


def gen_data_set(data, seq_max_len=50, negsample=0):
    data.sort_values("timestamp", inplace=True)
    
    train_set = []
    test_set = []
    for user_id, records in tqdm(data.groupby('user_id')):
        pos_movie_list = records['movie_id'].tolist()
        genres_list = records['genres'].tolist()

        if negsample > 0:
            item_ids = data['movie_id'].unique()
            item_id_genres_map = dict(zip(data['movie_id'].values, data['genres'].values))
            candidate_set = list(set(item_ids) - set(pos_movie_list)) # find those not in the user's selection
            neg_movie_list = np.random.choice(candidate_set, size=len(pos_movie_list) * negsample, replace=True)

        for i in range(1, len(pos_movie_list)):
            pos_movie_hist = pos_movie_list[:i]
            genres_hist = genres_list[:i]
            seq_len = min(i, seq_max_len)
            if i != len(pos_movie_list) - 1:
                train_set.append((
                    user_id, 
                    pos_movie_list[i], 
                    1, 
                    pos_movie_hist[::-1][:seq_len] + [0] * (seq_max_len - seq_len), 
                    seq_len, 
                    genres_hist[::-1][:seq_len] + [0] * (seq_max_len - seq_len),
                    genres_list[i],
                ))
                for neg_i in range(negsample):
                    train_set.append((
                        user_id, 
                        neg_movie_list[i * negsample + neg_i], 
                        0, 
                        pos_movie_hist[::-1][:seq_len] + [0] * (seq_max_len - seq_len), 
                        seq_len,
                        genres_hist[::-1][:seq_len] + [0] * (seq_max_len - seq_len), 
                        item_id_genres_map[neg_movie_list[i * negsample + neg_i]]
                    ))
            else:
                # one test sample for each user
                test_set.append((
                    user_id, 
                    pos_movie_list[i], 
                    1, 
                    pos_movie_hist[::-1][:seq_len] + [0] * (seq_max_len - seq_len), 
                    seq_len, 
                    genres_hist[::-1][:seq_len] + [0] * (seq_max_len - seq_len),
                    genres_list[i],
                ))

    random.shuffle(train_set)
    random.shuffle(test_set)

    print(len(train_set[0]), len(test_set[0]))

    return train_set, test_set


def gen_model_input(dataset, user_profile, seq_max_len):
    user_id = np.array([line[0] for line in dataset])
    movie_id = np.array([line[1] for line in dataset])
    label = np.array([line[2] for line in dataset])
    movie_seq = np.array([line[3] for line in dataset])
    hist_len = np.array([line[4] for line in dataset])
    genres_seq = np.array([line[5] for line in dataset])
    genres_id = np.array([line[6] for line in dataset])
    model_input = {"user_id": user_id, "movie_id": movie_id, "hist_movie_id": movie_seq,
                         "hist_genres": genres_seq,
                         "hist_len": hist_len, "genres": genres_id}

    for key in ["gender", "age", "occupation", "zip"]:
        model_input[key] = user_profile.loc[model_input['user_id']][key].values

    return model_input, label


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
    parser.add_argument("--negsample", type=int, default=10, help="num_pos_sample : num_neg_sample = 1 : negsample")
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
        encoding='latin-1',
    )
    rating_data = pd.read_csv(
        os.path.join(args.input_dir, "ratings.dat"),
        sep="::", 
        names=["user_id", "movie_id", "rating", "timestamp"],
        )

    tmp = pd.merge(rating_data, movie_data, how="left", on="movie_id", validate="many_to_one")
    data = pd.merge(tmp, user_data, how="left", on="user_id", validate="many_to_one")

    sparse_features = ["movie_id", "user_id",
                       "gender", "age", "occupation", "zip", "genres"]

    feature_max_idx = {}
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1
    
    user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')

    item_profile = data[["movie_id", "genres"]].drop_duplicates('movie_id')

    user_profile.set_index("user_id", inplace=True)

    train_set, test_set = gen_data_set(data, args.seq_len, args.negsample)
    train_model_input, train_label = gen_model_input(train_set, user_profile, args.seq_len)

