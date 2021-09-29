export PYTHONPATH=$PWD:$PYTHONPATH
set -aux

PRETRAINED_WEIGHT="./UNetmodel"
DATA_DIR="./CamVid"
SAVE_PATH="./test_results"

python test/test.py \
      --pretrained_path $PRETRAINED_WEIGHT \
      --data_dir $DATA_DIR \
      --save_path $SAVE_PATH \
