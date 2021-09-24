rm core.*

# export CUDA_VISIBLE_DEVICES=3

python3 eager_train.py \
    --model_load_dir ./checkpoints/merged_checkpoint \
    --model_save_dir ./checkpoints \
    --print_interval 1 \
    --deep_dropout_rate 0 \
    --max_iter 100

