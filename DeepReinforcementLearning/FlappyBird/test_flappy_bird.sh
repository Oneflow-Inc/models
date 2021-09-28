set -aux

PRETRAIN_MODEL_PATH="flappybird_pretrain_model"

if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/rl/flappybird_pretrain_model.zip
  unzip flappybird_pretrain_model.zip
fi

python3 test.py --saved_path $PRETRAIN_MODEL_PATH