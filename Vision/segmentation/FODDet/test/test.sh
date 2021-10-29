export PYTHONPATH=$PWD:$PYTHONPATH
set -aux

PRETRAINED_WEIGHT="./UNetmodel"
DATA_DIR="./CamVid"
SAVE_PATH="./test_results"

if [ ! -d "$PRETRAINED_WEIGHT" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/seg/FODDET/UNetmodel.zip
  unzip UNetmodel.zip
fi

python test/test.py \
      --pretrained_path $PRETRAINED_WEIGHT \
      --data_dir $DATA_DIR \
      --save_path $SAVE_PATH \
