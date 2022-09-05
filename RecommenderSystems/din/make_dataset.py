import pickle
import pandas as pd
import numpy as np
import argparse
import random
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.functions import rand, udf, lit, xxhash64
from pyspark.sql.types import StructField, StructType, FloatType, IntegerType, LongType, ArrayType

def to_df(file_path):
  with open(file_path, 'r') as fin:
    df = {}
    i = 0
    for line in fin:
      df[i] = eval(line)
      i += 1
    df = pd.DataFrame.from_dict(df, orient='index')
    return df


def build_maps(reviews_df, meta_df):
    def build_map(df, col_name):
        key = sorted(df[col_name].unique().tolist())
        m = dict(zip(key, range(len(key))))
        df[col_name] = df[col_name].map(lambda x: m[x])
        return m, key
    
    asin_map, asin_key = build_map(meta_df, 'asin')
    cate_map, cate_key = build_map(meta_df, 'categories')
    revi_map, revi_key = build_map(reviews_df, 'reviewerID')
    
    user_count, item_count, cate_count, example_count =\
        len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]
    print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
          (user_count, item_count, cate_count, example_count))
    
    meta_df = meta_df.sort_values('asin')
    meta_df = meta_df.reset_index(drop=True)
    reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
    reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
    reviews_df = reviews_df.reset_index(drop=True)
    reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
    
    cate_list = [meta_df['categories'][i] for i in range(len(asin_map))]
    cate_list = np.array(cate_list, dtype=np.int32)
    return reviews_df, cate_list, item_count


def build_dataset(reviews_df, item_count):
    max_len = 512 #max length is less than 512
    train_set = []
    test_set = []
    for reviewerID, hist in reviews_df.groupby('reviewerID'):
        pos_list = hist['asin'].tolist()
        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, item_count-1)
            return neg
        neg_list = [gen_neg() for i in range(len(pos_list))]
    
        for i in range(1, len(pos_list)):
            hist = pos_list[:i] + [0] * (512-i)
            if i != len(pos_list) - 1:
                train_set.append((hist, pos_list[i], 1.0, i))
                train_set.append((hist, neg_list[i], 0.0, i))
            else:
                test_set.append((hist, pos_list[i], 1.0, i))
                test_set.append((hist, neg_list[i], 0.0, i))
    
    #random.shuffle(train_set)
    #random.shuffle(test_set)
    return train_set, test_set


def build_by_spark():
    # start spark session
    conf = SparkConf()
    spark = SparkSession.builder.config(conf=conf).master("local[*]").getOrCreate()

    reviews_df = spark.createDataFrame(reviews_df)
    reviews_df = reviews_df.select(["reviewerID", xxhash64("asin").alias("asin"), "unixReviewTime"])

    meta_df = spark.createDataFrame(meta_df)
    meta_df = meta_df.select([xxhash64('asin').alias("asin"), xxhash64('categories').alias("categories")])

    reviews_df.printSchema()
    meta_df.printSchema()
    reviews_df.show()
    meta_df.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/data/xiexuan/dataset/amazon_elec/raw_data",
        help="path to amazon elec json files",
    )
    parser.add_argument("--output_dir", type=str, default="amazon_elec_parquet")
    args = parser.parse_args()

    print("read raw data")
    reviews_df = to_df(f'{args.input_dir}/reviews_Electronics_5.json')
    meta_df = to_df(f'{args.input_dir}/meta_Electronics.json')
    
    # keep used meta info
    meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
    meta_df = meta_df.reset_index(drop=True)
    
    reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
    meta_df = meta_df[['asin', 'categories']]
    meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])

    print("build_maps")
    reviews_df, cate_list, item_count = build_maps(reviews_df, meta_df)
    print("build_dataset")
    train_set, test_set = build_dataset(reviews_df, item_count)

    print(len(train_set), len(test_set))
    conf = SparkConf()
    conf.set("spark.driver.memory", f"128g")
    conf.set("spark.local.dir", "/data/xiexuan/tmp_spark")
    spark = SparkSession.builder.config(conf=conf).master("local[*]").getOrCreate()

    schema = StructType([       
        StructField('item_hist', ArrayType(IntegerType()), True),
        StructField('target', IntegerType(), True),
        StructField('label', FloatType(), True),
        StructField('seq_len', IntegerType(), True),
    ])


    train_df = spark.createDataFrame(train_set, schema=schema)
    test_df = spark.createDataFrame(test_set, schema=schema)

    train_df.orderBy(rand()).write.mode("overwrite").parquet(f"{args.output_dir}/train")
    test_df.write.mode("overwrite").parquet(f"{args.output_dir}/test")

    train_df.printSchema()
    test_df.show()
