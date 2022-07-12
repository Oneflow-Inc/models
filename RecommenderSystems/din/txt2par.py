import pandas as pd
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--part", type=int, required=True)

cols = ['item_his', 'cat_his', 'tar_id', 'tar_cat', 'label']
dataset_list = ['train', 'test', 'val']
args = parser.parse_args()
part = args.part


for dataset in dataset_list:
    tmp_df = pd.read_csv('./data_big/{}/{}_cat801.txt'.format(dataset, dataset),header=None,sep=';')
    tmp_df = tmp_df.sample(frac=1).reset_index(drop='index')
    len_part = tmp_df.shape[0] // part
    for i in range(part - 1):
        tmp_part = tmp_df[i * len_part : (i + 1) * len_part]
        tmp_part.columns = cols
        tmp_part.to_parquet('./data_big/{}/{}_cat801_{}.parquet'.format(dataset, dataset, i + 1))
    # if tmp_df.shape[0] % part != 0:
    tmp_part = tmp_df[(part - 1) * len_part :]
    tmp_part.columns = cols
    tmp_part.to_parquet('./data_big/{}/{}_cat801_{}.parquet'.format(dataset, dataset, part))
