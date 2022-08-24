model=bert-tiny-uncased

# In domain_sentiment_data, genre is one of ["books", "dvd", "electronics", "kitchen"]
genre=books

python meta_student_distill.py \
--mode=train \
--tables=data/SENTI/train_with_weights.tsv,data/SENTI/dev_standard.tsv \
--input_schema=guid:str:1,text_a:str:1,text_b:str:1,label:str:1,domain:str:1,weight:str:1 \
--first_sequence=text_a \
--second_sequence=text_b \
--label_name=label \
--label_enumerate_values=positive,negative \
--checkpoint_dir=./tmp/$genre/meta_student_pretrain_for_review/ \
--learning_rate=3e-5  \
--epoch_num=5  \
--random_seed=42 \
--logging_steps=20 \
--sequence_length=128 \
--micro_batch_size=16 \
--app_name=text_classify \
--user_defined_parameters="
      pretrain_model_name_or_path=$model
      student_config_path=./bert-tiny-uncased-oneflow
      teacher_model_path=./tmp/meta_teacher_review/
      teacher_config_path=./bert-base-uncased-oneflow
      domain_loss_weight=0.5
      distill_stage=first
      genre=$genre
      T=2
      "