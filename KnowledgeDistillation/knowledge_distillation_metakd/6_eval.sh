# 3. Evalute
Student_model_path=./tmp/books/meta_student_fintune
python main_evaluate.py \
--mode evaluate \
--tables=data/SENTI/train_with_weights.tsv,data/SENTI/dev_standard.tsv \
--input_schema=guid:str:1,text_a:str:1,text_b:str:1,label:str:1,domain:str:1,weight:str:1 \
--first_sequence=text_a \
--label_name=label \
--label_enumerate_values=positive,negative \
--checkpoint_dir=./bert-tiny-uncased \
--learning_rate=3e-5  \
--epoch_num=1  \
--random_seed=42 \
--logging_steps=20 \
--sequence_length=128 \
--micro_batch_size=16 \
--app_name=text_classify \
--user_defined_parameters="
    pretrain_model_name_or_path=$Student_model_path
    genre=$genre
    student_config_path=./bert-tiny-uncased-oneflow
    student_model_path=$Student_model_path
    "