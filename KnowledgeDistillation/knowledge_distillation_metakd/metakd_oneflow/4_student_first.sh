model=bert-tiny-uncased

# In domain_sentiment_data, genre is one of ["books", "dvd", "electronics", "kitchen"]
genre=books
# cd ${cur_path}

# 1. Distillation pretrain
# 进行分布式训练
# nproc_per_node：每个节点上的 GPU 数目
# nnodes：节点数目（机器数目）
# 
# DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 7021"
# Pretrained distillation
# python -m torch.distributed.launch $DISTRIBUTED_ARGS meta_student_distill.py \
python meta_student_distill.py \
--mode train \
--tables=data/SENTI/train_with_weights.tsv,data/SENTI/dev_standard.tsv \
--input_schema=guid:str:1,text_a:str:1,text_b:str:1,label:str:1,domain:str:1,weight:str:1 \
--first_sequence=text_a \
--second_sequence=text_b \
--label_name=label \
--label_enumerate_values=positive,negative \
--checkpoint_dir=./tmp/$genre/meta_student_pretrain/ \
--learning_rate=3e-5  \
--epoch_num=3  \
--random_seed=42 \
--logging_steps=20 \
--sequence_length=128 \
--micro_batch_size=16 \
--app_name=text_classify \
--user_defined_parameters="
      pretrain_model_name_or_path=$model
      teacher_model_path=./tmp/meta_teacher/
      domain_loss_weight=0.5
      distill_stage=first
      genre=$genre
      T=2
      "