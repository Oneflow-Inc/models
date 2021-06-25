CW=100000
LR=0.001
EPOCH=0
CKPT=65000
MODEL="checkpoints/SKETCHCROP-1_CW_40000_lr_0.001ckpt_sketch_epoch0_4000"
#MODEL="saved_models/vgg16-sketch/"
IMAGE="oneflow.png"
CONTENT="images/content-images/$IMAGE"
OUTPUT="oneflow-bottle-CW40000-4000.jpg"
CUDA=1

python3 neural_style/neural_style.py eval \
    --model $MODEL \
    --content-image $CONTENT \
    --output-image $OUTPUT \
    --cuda $CUDA