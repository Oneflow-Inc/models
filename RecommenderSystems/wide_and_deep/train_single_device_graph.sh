MASTER_ADDR=127.0.0.1
DATA_DIR=/dataset/wdl_ofrecord/ofrecord
EMBD_SIZE=2322444

# batch_size=32 64 128 ...
python3 train.py \
        --batch_size 256 \
        --data_dir $DATA_DIR \
        --print_interval 100 \
        --deep_dropout_rate 0.5 \
        --max_iter 1100 \
        --hidden_units_num 2 \
        --wide_vocab_size $EMBD_SIZE \
        --deep_vocab_size $EMBD_SIZE \
        --gpu_num_per_nod 1 \
        --node_ips $MASTER_ADDR \
        --eval_batchs 0 \
        --execution_mode 'graph' 2>&1 | tee logfile.log

echo -e "\n"

python3 extract_time.py --log_file logfile.log

echo -e "\n"
