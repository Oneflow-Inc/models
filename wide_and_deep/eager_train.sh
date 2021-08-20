rm core.*

# export CUDA_VISIBLE_DEVICES=3

python3 eager_train.py \
    --model_load_dir ./checkpoints/merged_checkpoint \
    --model_save_dir ./checkpoints \
    --print_interval 1 \
    --deep_dropout_rate 0 \
    --max_iter 100
    # --save_initial_model \
    # --model_load_dir ./checkpoints/initial_checkpoint \
    # --model_load_dir /home/xiexuan/sandbox/OneFlow-Benchmark/ClickThroughRate/WideDeepLearning/baseline_checkpoint \
