import pandas as pd

train = pd.read_csv('./data/train/train.txt',header=None,sep=';')
test = pd.read_csv('./data/val/test.txt',header=None,sep=';')
cols = ['item_his', 'cat_his', 'tar_id', 'tar_cat', 'label']
train.columns = cols
test.columns = cols

train.to_parquet('./data/train/train.parquet')
test.to_parquet('./data/val/test.parquet')