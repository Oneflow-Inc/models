rm -rf /home/zhengzekang/models/RecommenderSystems/deepfm/deepfm_persistent_file/0-1/*

# export ONEFLOW_DEBUG_MODE=1
export CUDA_VISIBLE_DEVICES=1
DEVICE_NUM_PER_NODE=1
DATA_DIR=/data/xiexuan/dataset/din_pkl

PERSISTENT_PATH=/home/zhengzekang/models/RecommenderSystems/deepfm/deepfm_persistent_file

python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    din_train_eval_pkl.py \
      --data_dir $DATA_DIR \
      --batch_size 32 \
      --max_len 512 \
      --train_batches 1630490 \
      --learning_rate 1.0 \
      --save_model_after_each_eval \
      --persistent_path $PERSISTENT_PATH \
      | tee run.log
      #--save_model_after_each_eval \
      #--model_save_dir ckpt \
      # --save_initial_model \