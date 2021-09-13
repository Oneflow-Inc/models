set -aux

CHECKPOINT_PATH="checkpoints"
if [ ! -d "$CHECKPOINT_PATH" ]; then
    mkdir $CHECKPOINT_PATH
fi

LEARNING_RATE=1e-6
OPTIMIZER="adam"
EPOCH=2000000
BATCH_SIZE=32
REPLAY_MEMORY_SIZE=50000


python3 train.py \
    --save_checkpoint_path $CHECKPOINT_PATH \
    --lr $LEARNING_RATE \
    --optimizer $OPTIMIZER \
    --num_iters $EPOCH \
    --batch_size $BATCH_SIZE \
    --replay_memory_size $REPLAY_MEMORY_SIZE \
    # --load_checkpoint $LOAD_CHECKPOINT