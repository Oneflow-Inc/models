rm core.*

# export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

python3 eager_train.py \
    --model_save_dir ./checkpoints \
    --print_interval 1 \
    --deep_dropout_rate 0 \
    --max_iter 10000 \
    --eval_batchs 0 \
    --data_dir /data/wdl_ofrecord \
    --batch_size 32
    # --save_initial_model \
    # --model_load_dir ./checkpoints/initial_checkpoint \
    # --model_load_dir /home/xiexuan/sandbox/OneFlow-Benchmark/ClickThroughRate/WideDeepLearning/baseline_checkpoint \
