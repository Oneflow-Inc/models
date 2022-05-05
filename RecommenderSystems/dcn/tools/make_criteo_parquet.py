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
import numpy as np

import numpy as np
from collections import Counter, OrderedDict
import pandas as pd
import io
import pickle
import os
import logging
import json
from collections import defaultdict
from collections import Counter
import itertools
import numpy as np
import pandas as pd
import pickle
import os
import sklearn.preprocessing as sklearn_preprocess

import numpy as np
from torch.utils import data
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


class Tokenizer(object):
    def __init__(
        self,
        na_value=None,
        min_freq=1,
        oov_token=0,
        max_len=0,
        padding="pre",
    ):
        self._na_value = na_value
        self._min_freq = min_freq
        self.oov_token = oov_token  # use 0 for __OOV__
        self.word_counts = Counter()
        self.vocab = dict()
        self.vocab_size = 0  # include oov and padding
        self.max_len = max_len
        self.padding = padding

    def fit_on_texts(self, texts, use_padding=True):
        tokens = list(texts)
        if self._na_value is not None:
            tokens = [tk for tk in tokens if tk != self._na_value]
        self.word_counts = Counter(tokens)
        words = [token for token, count in self.word_counts.items() if count >= self._min_freq]
        self.word_counts.clear()  # empty the dict to save memory
        self.vocab = dict((token, idx) for idx, token in enumerate(words, 1 + self.oov_token))
        self.vocab["__OOV__"] = self.oov_token
        if use_padding:
            self.vocab["__PAD__"] = (
                len(words) + self.oov_token + 1
            )  # use the last index for __PAD__
        self.vocab_size = len(self.vocab) + self.oov_token

    def encode_category(self, categories):
        category_indices = [self.vocab.get(x, self.oov_token) for x in categories]
        return np.array(category_indices)



class FeatureMap(object):
    def __init__(self, dataset_id):
        self.dataset_id = dataset_id
        self.num_fields = 0
        self.num_features = 0
        self.feature_len = 0
        self.feature_specs = OrderedDict()

    def set_feature_index(self):
        logging.info("Set feature index...")
        idx = 0
        for feature, feature_spec in self.feature_specs.items():
            self.feature_specs[feature]["index"] = idx
            idx += 1
        self.feature_len = idx

    def save(self, json_file):
        logging.info("Save feature_map to json: " + json_file)
        if not os.path.exists(os.path.dirname(json_file)):
            os.makedirs(os.path.dirname(json_file))
        feature_map = OrderedDict()
        feature_map["dataset_id"] = self.dataset_id
        feature_map["num_fields"] = self.num_fields
        feature_map["num_features"] = self.num_features
        feature_map["feature_len"] = self.feature_len
        feature_map["feature_specs"] = self.feature_specs
        with open(json_file, "w") as fd:
            json.dump(feature_map, fd, indent=4)


class FeatureEncoder(object):
    def __init__(
        self,
        feature_cols=[],
        label_col={},
        dataset_id=None,
        data_root="../data/",
        **kwargs
    ):
        logging.info("Set up feature encoder...")
        self.data_dir = os.path.join(data_root, dataset_id)     
        self.pickle_file = os.path.join(self.data_dir, "feature_encoder.pkl")   
        self.json_file = os.path.join(self.data_dir, "feature_map.json")     
        self.feature_cols = self._complete_feature_cols(feature_cols)       
        self.label_col = label_col                                                                                
        self.feature_map = FeatureMap(dataset_id)
        self.encoders = dict()

    def _complete_feature_cols(self, feature_cols):
        full_feature_cols = []
        for col in feature_cols:
            name_or_namelist = col["name"]
            if isinstance(name_or_namelist, list):
                for _name in name_or_namelist:
                    _col = col.copy()
                    _col["name"] = _name
                    full_feature_cols.append(_col)
            else:
                full_feature_cols.append(col)
        return full_feature_cols

    def read_csv(self, data_path):
        logging.info("Reading file: " + data_path)
        all_cols = self.feature_cols + [self.label_col]
        dtype_dict = dict(
            (x["name"], eval(x["dtype"]) if isinstance(x["dtype"], str) else x["dtype"])
            for x in all_cols
        )
        ddf = pd.read_csv(data_path, dtype=dtype_dict, memory_map=True)
        return ddf

    def _preprocess(self, ddf):
        logging.info("Preprocess feature columns...")
        all_cols = [self.label_col] + self.feature_cols[::-1]
        for col in all_cols:
            name = col["name"]
            if name in ddf.columns and ddf[name].isnull().values.any():
                ddf[name] = self._fill_na(col, ddf[name])
            if "preprocess" in col and col["preprocess"] != "":
                preprocess_fn = getattr(self, col["preprocess"])
                ddf[name] = preprocess_fn(ddf, name)
        active_cols = [self.label_col["name"]] + [
            col["name"] for col in self.feature_cols if col["active"]
        ]
        ddf = ddf.loc[:, active_cols]
        return ddf

    def _fill_na(self, col, series):
        na_value = col.get("na_value")
        if na_value is not None:
            return series.fillna(na_value)
        elif col["dtype"] == "str":
            return series.fillna("")
        else:
            raise RuntimeError("Feature column={} requires to assign na_value!".format(col["name"]))

    def fit(self, train_data, min_categr_count=1, num_buckets=10, **kwargs):
        ddf = self.read_csv(train_data)
        ddf = self._preprocess(ddf)
        logging.info("Fit feature encoder...")
        self.feature_map.num_fields = 0
        for col in self.feature_cols:
            if col["active"]:
                logging.info("Processing column: {}".format(col))
                name = col["name"]
                self.fit_feature_col(
                    col, ddf, min_categr_count=min_categr_count, num_buckets=num_buckets
                )
                self.feature_map.num_fields += 1
        self.feature_map.set_feature_index()
        self.feature_map.save(self.json_file)
        logging.info("Set feature encoder done.")

    def fit_feature_col(self, feature_column, ddf, min_categr_count=1, num_buckets=10):
        name = feature_column["name"]
        feature_type = feature_column["type"]
        feature_source = feature_column.get("source", "")
        self.feature_map.feature_specs[name] = {"source": feature_source, "type": feature_type}
        if "min_categr_count" in feature_column:
            min_categr_count = feature_column["min_categr_count"]
        self.feature_map.feature_specs[name]["min_categr_count"] = min_categr_count
        feature_values = ddf[name].values
        if feature_type == "categorical":
            tokenizer = Tokenizer(
                    min_freq=min_categr_count, na_value=feature_column.get("na_value", "")
                )
            tokenizer.fit_on_texts(feature_values, use_padding=False)
            self.encoders[name + "_tokenizer"] = tokenizer
            self.feature_map.num_features += tokenizer.vocab_size
            self.feature_map.feature_specs[name]["vocab_size"] = tokenizer.vocab_size
        else:
            raise NotImplementedError("feature_col={}".format(feature_column))

    def transform(self, ddf):
        ddf = self._preprocess(ddf)
        logging.info("Transform feature columns...")
        data_arrays = []
        for feature, feature_spec in self.feature_map.feature_specs.items():
            feature_type = feature_spec["type"]
            if feature_type == "categorical":
                encoder = feature_spec.get("encoder", "")
                if encoder == "":
                    data_arrays.append(
                        self.encoders.get(feature + "_tokenizer").encode_category(
                            ddf.loc[:, feature].values))
                else:
                    raise NotImplementedError
            else :
                raise NotImplementedError
              
        label_name = self.label_col["name"]
        if ddf[label_name].dtype != np.float64:
            ddf.loc[:, label_name] = ddf.loc[:, label_name].apply(lambda x: float(x))
        data_arrays.append(ddf.loc[:, label_name].values)  # add the label column at last
        data_arrays = [item.reshape(-1, 1) if item.ndim == 1 else item for item in data_arrays]
        data_array = np.hstack(data_arrays)
        return data_array

    def convert_to_bucket(self, df, col_name):
        def _convert_to_bucket(value):
            if value > 2:
                value = int(np.floor(np.log(value) ** 2))
            else:
                value = int(value)
            return value

        return df[col_name].map(_convert_to_bucket).astype(int)


def load_config(config_dir, dataset_id):
    params = dict()
    params["dataset_id"] = dataset_id
    dataset_configs = glob.glob(os.path.join(config_dir, "dataset_config.yaml"))
    for config in dataset_configs:
        with open(config, "r") as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if dataset_id in config_dict:
                params.update(config_dict[dataset_id])
                break
    return params


class DataIO(object):
    def __init__(self, feature_encoder):
        self.feature_encoder = feature_encoder

    def load_data(self, data_path):
        ddf = self.feature_encoder.read_csv(data_path)
        data_array = self.feature_encoder.transform(ddf)
        return data_array


def data_generator(
    feature_encoder,
    train_data=None,
    valid_data=None,
    test_data=None,
    **kwargs
):
    data_io = DataIO(feature_encoder)
    train_array = data_io.load_data(train_data)
    valid_array = data_io.load_data(valid_data)
    test_array = data_io.load_data(test_data)

    return train_array, valid_array, test_array


def make_array_to_parquet(spark, feature_map, data_array, output_dir, part_num=None, shuffle=False):
    # cols: 39 features + 1 label
    X = data_array[:, :-1]
    label = data_array[:, -1].reshape(-1, 1)

    print("start transforming data!")
    total_prev_vocab = 0
    for key in feature_map["feature_specs"].keys():
        X[:, feature_map["feature_specs"][key]["index"]] += total_prev_vocab
        total_prev_vocab += float(feature_map["feature_specs"][key]["vocab_size"])

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

    del X, label, data_array, data_pandas, df
    return num_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/yuanziyang/yzywork/dcn_for_pr/tools", help="The config directory.")
    parser.add_argument("--dataset_id", type=str, default="criteo_x4_001")
    # parser.add_argument("--output_dir", type=str, default="/home/yuanziyang/yzywork/dcn_for_pr/Criteo_sample/Criteo_parquet")
    parser.add_argument("--output_dir", type=str, default="/home/yuanziyang/yzywork/dcn_for_pr/Criteo/Criteo_parquet")
    parser.add_argument("--spark_tmp_dir", type=str, default=None)
    parser.add_argument("--spark_driver_memory_gb", type=int, default=1024)
    parser.add_argument("--export_dataset_info", action="store_true", help="export dataset infomation or not")
    args = vars(parser.parse_args())
    args["export_dataset_info"] = True


    params = load_config(args["config"], args["dataset_id"])

    feature_encoder = FeatureEncoder(**params)
    feature_encoder.fit(**params)


    train_array, valid_array, test_array = data_generator(feature_encoder, **params)

    data_dir = os.path.join(params["data_root"], params["dataset_id"])
    feature_map_json = os.path.join(data_dir, "feature_map.json") 

    print("Start Loading feature map!")
    with io.open(feature_map_json, "r", encoding="utf-8") as fd:
        feature_map = json.load(fd, object_pairs_hook=OrderedDict)
    vocab_size_array = [feature_map["feature_specs"][key]["vocab_size"] for key in feature_map["feature_specs"].keys()]
    print("Loading feature map done!")

    # start spark session
    conf = SparkConf()
    spark_driver_memory_gb = args["spark_driver_memory_gb"]
    conf.set("spark.driver.memory", f"{spark_driver_memory_gb}g")
    conf.set("spark.local.dir", args["spark_tmp_dir"])
    conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark = SparkSession.builder.config(conf=conf).master("local[*]").getOrCreate()

    print("start making test parquet data!")
    test_output_dir = os.path.join(args["output_dir"], "test")
    test_count = make_array_to_parquet(
        spark, feature_map, test_array, test_output_dir, part_num=256, shuffle=False
    )
    del test_array

    print("start making val parquet data!")
    val_output_dir = os.path.join(args["output_dir"], "val")
    val_count = make_array_to_parquet(
        spark, feature_map, valid_array, val_output_dir, part_num=256, shuffle=False
    )
    del valid_array

    print("start making train parquet data!")
    train_output_dir = os.path.join(args["output_dir"], "train")
    train_count = make_array_to_parquet(
        spark, feature_map, train_array, train_output_dir, part_num=1024, shuffle=True
    )
    del train_array

    if args["export_dataset_info"]:
        df = spark.read.parquet(train_output_dir, test_output_dir, val_output_dir)
        table_size_array = [df.select(f"I{i}").distinct().count() for i in range(1, 14)] + [
            df.select(f"C{i}").distinct().count() for i in range(1, 27)
        ]
        print("table_size_array:", table_size_array)
        print("vocab_size_array:", vocab_size_array)
        with open(os.path.join(args["output_dir"], "README.md"), "w") as f:
            f.write("## number of examples:\n")
            f.write(f"train: {train_count}\n")
            f.write(f"test: {test_count}\n")
            f.write(f"val: {val_count}\n\n")
            f.write("## table size array\n")
            f.write("table_size_array = [")
            f.write(", ".join([str(i) for i in table_size_array]))
            f.write("]\n\n")
            f.write("## vocab size array\n")
            f.write("vocab_size_array = [")
            f.write(", ".join([str(i) for i in vocab_size_array]))
            f.write("]\n")
