
python train.py \
    --model_save_dir="./output/model_save" \
    --image_save_name="./output/images/compare_result.jpg" \
    --load_teacher_checkpoint_dir="./output/model_save/teacher" \
    --model_type="compare"\
    --epochs=10 \
    --batch_size=128 \
    --temperature=5.0 \
    --alpha=0.7 \
    

