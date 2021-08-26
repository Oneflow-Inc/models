EVAL_MODE='torch'
MODEL='alexnet'

CUDA_VISIBLE_DEVICES=1 python eval/eval_torch_model.py --eval_mode $EVAL_MODE --model $MODEL