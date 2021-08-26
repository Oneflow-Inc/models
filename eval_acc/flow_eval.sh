EVAL_MODE='flow'
MODEL='alexnet'

CUDA_VISIBLE_DEVICES=1 python eval/eval_flow_model.py --eval_mode $EVAL_MODE --model $MODEL