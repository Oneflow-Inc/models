set -aux

PRETRAIN_MODEL_PATH="./checkpoints/epoch_19_val_acc_0.732143"
IMAGE_PATH="imagenette2/val/n01440764/ILSVRC2012_val_00009111.JPEG"
QUANTIZATION_BIT=8
QUANTIZATION_SCHEME="symmetric"
QUANTIZATION_FORMULA="google"
PER_LAYER_QUANTIZATION=True


if [ ! -d "data" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/data.tar.gz
  tar zxf data.tar.gz
fi

python3 quantization_infer.py \
    --model_path $PRETRAIN_MODEL_PATH\
    --image_path $IMAGE_PATH\
    --quantization_bit $QUANTIZATION_BIT \
    --quantization_scheme $QUANTIZATION_SCHEME \
    --quantization_formula $QUANTIZATION_FORMULA \
    --per_layer_quantization $PER_LAYER_QUANTIZATION
 