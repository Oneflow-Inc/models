rm -rf ./data_big/train/*.parquet
rm -rf ./data_big/test/*.parquet
rm -rf ./data_big/val/*.parquet
python txt2par.py --part 800
