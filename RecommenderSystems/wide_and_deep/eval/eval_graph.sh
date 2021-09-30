
python3 eval.py \
        --deep_vocab_size 2322444\
        --wide_vocab_size 2322444\
        --hidden_units_num 2\
        --deep_embedding_vec_size 16\
        --batch_size 512\
        --model_save_dir ./checkpoints \
        --print_interval 100 \
        --deep_dropout_rate 0 \
        --max_iter 1000 \
        --eval_name 'graph'

