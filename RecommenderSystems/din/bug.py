import pandas as pd

train = pd.read_csv('./data/train/train.txt',header=None,sep=';')
test = pd.read_csv('./data/val/test.txt',header=None,sep=';')
cols = ['item_his', 'cat_his', 'tar_id', 'tar_cat', 'label']
# train.columns = cols
# test.columns = cols
test = test.sample(frac=1)  
test_ = test[:test.shape[0] // 2]
val_ = test[test.shape[0] // 2:]

train.columns = cols
test_.columns = cols
val_.columns = cols
print(test_.head())
print(val_.head())
train.to_parquet('./data/train/train.parquet')
test_.to_parquet('./data/test/test.parquet')
val_.to_parquet('./data/val/val.parquet')