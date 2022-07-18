## Meta Knowledge Distillation
Oneflow implement for [Meta Knowledge Distillation（Meta-KD）](https://arxiv.org/pdf/2012.01266.pdf") algorithm

---

## Meta-KD：
Pre-training language model for knowledge distillation can greatly improve the execution efficiency of the model without reducing the effect of the model as much as possible. The previous method is to follow the teacher-student method to achieve model distillation, but they ignore the domain knowledge learned by the teacher model The problem of deviations from the student model. Based on this, Meta-KD proposes to adopt a meta-learning method, let the teacher model learn meta-knowledge on different domains first, obtain meta-teacher, and then let the student model learn meta-teacher's prior knowledge, trying to make the student model also With meta-knowledge.
Taking the text classification task as an example, the algorithm flow is roughly as follows:
- First obtain the N-way K-shot classification data, obtain the prototypical embedding of each class according to each data set, and calculate the prototypical score of each sample;
- Train the meta-teacher model, use meta-learning method to learn domain-knowledge, including prototype and domain corruption, etc.;
- According to the trained meta-teacher model, perform inference on the training set to obtain the prior knowledge of each sample, including the attention values corresponding to the sample, the representation vector of the last layer, and the predicted logits values;
- For training meta-student model, the cross-entropy loss function is used for logits values, and the average MSE is used for attention values and representation vectors;

## Data Acquisition
Take Amazon review evaluation task as an example, downloadable corpus:
The Amazon Review dataset can be found in this [link](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html)


## Experiment Settings

#### Step1：Generating prototype embedding and prototypical score

```shell
python3 preprocess.py \
        --task_name senti \
        --model_load_dir uncased_L-12_H-768_A-12_oneflow \
        --data_dir data/SENTI/ \
        --num_epochs 4 \
        --seed 42 \
        --seq_length=128 \
        --train_example_num 6480 \
        --vocab_file uncased_L-12_H-768_A-12/vocab.txt \
        --resave_ofrecord
```

After execution, the corresponding ofrecord data will be obtained.


#### Step2：Training the Meta-teacher Model

This part needs to use the teacher model to perform meta-learning on the target data:

```shell
python3 meta_teacher.py \
        --task_name senti \
        --model_load_dir uncased_L-12_H-768_A-12_oneflow \
        --data_dir data/SENTI/ \
        --num_epochs 63 \
        --seed 42 \
        --seq_length=128 \
        --train_example_num 6480 \
        --eval_example_num 720 \
        --batch_size_per_device 24 \
        --eval_batch_size_per_device 48 \
        --eval_every_step_num 100 \
        --vocab_file uncased_L-12_H-768_A-12/vocab.txt \
        --learning_rate 5e-5 \
        --resave_ofrecord \
        --do_train \
        --do_eval
```
After training, the meta-teacher model will be obtained, and the model will be saved in the output directory

#### Step3：Training the Meta Distillation

First, on the training set, get meta-teacher's soft-label, attention, embedding and other parameters, and save them locally

```shell
python3 meta_teacher_eval.py \
        --task_name senti \
        --model_load_dir output/model_save-2021-09-26-15:31:15/snapshot_best_mft_model_senti_dev_0.8691358024691358 \
        --data_dir data/SENTI/ \
        --num_epochs 4 \
        --seed 42 \
        --seq_length=128 \
        --train_example_num 6480 \
        --eval_batch_size_per_device 1 \
        --vocab_file uncased_L-12_H-768_A-12/vocab.txt \
        --resave_ofrecord
```

Then train the student model and perform distillation:

```shell
python3 meta_distill.py 
        --task_name senti \
        --student_model uncased_L-12_H-768_A-12_oneflow \
        --teacher_model output/model_save-2021-09-26-15:31:15/snapshot_best_mft_model_senti_dev_0.8691358024691358 \
        --data_dir data/SENTI/ \
        --num_epochs 63 \
        --seed 42 \
        --seq_length=128 \
        --train_example_num 6480 \
        --eval_example_num 720 \
        --batch_size_per_device 24 \
        --eval_batch_size_per_device 48 \
        --eval_every_step_num 100 \
        --vocab_file uncased_L-12_H-768_A-12/vocab.txt \
        --learning_rate 5e-5 \
        --resave_ofrecord \
        --do_train \
        --do_eval
```
