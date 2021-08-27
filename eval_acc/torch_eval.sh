EVAL_MODE='torch'
MODEL='alexnet'
CHECKPOINT_PATH="/data/rentianhe/code/new_models/models/eval_acc/weight/torch/alexnet-owt-4df8aa71.pth"

CUDA_VISIBLE_DEVICES=1 python eval/eval_torch_model.py \
                              --eval_mode $EVAL_MODE \
                              --model $MODEL \
                              --checkpoint_path $CHECKPOINT_PATH 