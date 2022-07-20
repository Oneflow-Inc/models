python3 preprocess.py \
        --task_name senti \
        --model_load_dir uncased_L-12_H-768_A-12_oneflow \
        --data_dir data/SENTI/ \
        --num_epochs 4 \
        --seed 42 \
        --seq_length=128 \
        --train_example_num 6480 \
        --vocab_file uncased_L-12_H-768_A-12/vocab.txt \
        --resave_ofrecord