#! /bin/sh
INPUT_DIR=./Frappe_x1
OUTPUT_DIR=./Frappe_x1_parquet

python frappe-parquet.py \
    --input_dir $INPUT_DIR\
    --output_dir $OUTPUT_DIR\
    --export_dataset_info