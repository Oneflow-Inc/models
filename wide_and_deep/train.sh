rm core.*

# export CUDA_VISIBLE_DEVICES=3

python3 train.py \
    --model_load_dir ./checkpoints/initial_checkpoint \
    --print_interval 1 \
    --max_iter 1000

    # --model_save_dir ./checkpoints \
    # --save_initial_model \
