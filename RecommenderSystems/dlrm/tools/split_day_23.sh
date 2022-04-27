# split day_23 to test.csv and val.csv
src_dir="/workspace/dataset/criteo1t/raw"
tmp_dir="/workspace/tmp_spark"

mkdir -p /workspace/tmp_spark

# total 178274637, test 89137319, val 89137318
head -n 89137319 $src_dir/day_23 > $tmp_dir/test.csv
tail -n +89137320 $src_dir/day_23 > $tmp_dir/val.csv
