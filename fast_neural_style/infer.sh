PRETRAIN_MODEL_PATH="style_models/"
MODEL="sketch" # choose from sketch, candy, mosaic, rain_princess, udnie

if [ ! -d "$PRETRAIN_MODEL_PATH" ]; then
  mkdir ${PRETRAIN_MODEL_PATH}
fi

if [ ! -d "${PRETRAIN_MODEL_PATH}${MODEL}_oneflow_model" ]; then
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/fast_neural_style/${MODEL}_oneflow.tar.gz
  tar zxf ${MODEL}_oneflow.tar.gz --directory ${PRETRAIN_MODEL_PATH}
fi

MODEL="${PRETRAIN_MODEL_PATH}${MODEL}_oneflow"
IMAGE="oneflow.png" # change this line
IMAGE_NAME=y=${IMAGE%%.*}
CONTENT="images/content-images/${IMAGE}"
OUTPUT="images/output-images/${IMAGE_NAME}.jpg"
CUDA=1

python3 neural_style/neural_style.py eval \
    --model $MODEL \
    --content-image $CONTENT \
    --output-image $OUTPUT \
    --cuda $CUDA