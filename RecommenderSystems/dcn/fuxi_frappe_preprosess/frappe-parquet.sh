#! /bin/sh
INPUT_ACCUM_DIR=/home/yuanziyang/yzywork/dcn-test-dir/frappe_temp_accum
OUTPUT_PARQUET_DIR=/home/yuanziyang/yzywork/dcn-test-dir/frappe_temp_parquet

python process2.py \
    --input_accum_dir $INPUT_ACCUM_DIR\
    --output_parquet_dir $OUTPUT_PARQUET_DIR\
    --export_dataset_info