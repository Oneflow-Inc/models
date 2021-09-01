LEARNING_RATE=0.001
EPOCH=50
BATCH_SIZE=8

 python train.py \
      --DATA_DIR $DATASET_PATH \
      --save_checkpoint_path $CHECKPOINT_PATH \
      --lr $LEARNING_RATE \
      --epoch $EPOCH\
      --batch_size $BATCH_SIZE\
