#usr/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py \
    --train_dataset "data/corpus.small" \
    --test_dataset "data/corpus.small" \
    --vocab_file "gpt2-vocab.json" \
    --merges_file "gpt2-merge.txt" \
    --output_path "outputs" \
    --restore_file "gpt2_oneflow_model" \
    --seq_len 128 \
    --batch_size 4 \
    --epochs 20 \
    --num_workers 1 \
    --lr 3e-4 \
    --adam_weight_decay 0.01 \
    --adam_beta1 0.98 \
    --adam_beta2 0.999 \
    --warmup_steps 0 \
    --accumulate_gradient_steps 1;