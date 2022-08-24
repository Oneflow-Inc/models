# Meta-KD
Implementation of the Meta-KD algorithm with [OneFlow](https://github.com/Oneflow-Inc/oneflow#install-with-pip-package).
The reference papers are: https://arxiv.org/abs/2012.01266v1. This project refers to the implementation of this algorithm by [EasyNLP](https://github.com/alibaba/EasyNLP/tree/master/examples/knowledge_distillation/metakd). Complete the migration by simply modifying the statement "import oneflow as torch".
---
## Knowledge Distillation：
The distillation of the pre training language model often only focuses on the knowledge of a single domain, and the student model can only obtain knowledge from the teacher model in the corresponding domain. Knowledge distillation can enable the student model to acquire knowledge from multiple teachers from different fields or cross field teachers, and then help the student model training in the target field. However, this method may transfer some non migrating knowledge from other fields, which is irrelevant to the current field, thus causing the model to decline. Cross task knowledge distillation obtains transferable knowledge of multiple domains through meta learning, and improves the generalization performance of teacher model on cross domain knowledge to improve the performance of student model.

The meta KD algorithm is different from the existing cross task knowledge distillation. It uses the idea of meta learning for reference. First, a meta teacher is trained on multiple different domain datasets to obtain the knowledge of the transferability of multiple domains. On the basis of this meta teacher, the model is distilled to the student model based on specific tasks, and better results are achieved. The algorithm idea of meta KD algorithm is shown in the following figure:
![image.png](https://cdn.nlark.com/yuque/0/2022/png/2556002/1647500661600-17f5d7c5-eafc-43e6-b4a5-3c51156b12e9.png#clientId=u0e51cb8b-d6a9-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=306&id=u1b7119ae&margin=%5Bobject%20Object%5D&name=image.png&originHeight=782&originWidth=1034&originalType=binary&ratio=1&rotation=0&showTitle=false&size=330592&status=done&style=none&taskId=ud1819c79-eeb8-48cf-a94d-3fb50714266&title=&width=304)

In the implementation of the algorithm, firstly, based on the training data in different fields, the meta teacher is trained. Due to the different portability of data in different fields, we use the method based on class centroid to calculate the weight of each data (i.e. prototype score in the figure below), indicating the portability of this data to other fields. Generally speaking, the smaller the domain characteristics, the greater the weight. Meta teacher performs weighted hybrid training on domain data. After the training of meta teacher, we distill this model into the data of a specific field, and fully consider the combination of multiple loss functions. In addition, since the meta teacher does not necessarily have good performance in all fields of data, we used the domain expert weight to measure the confidence of the meta teacher in the correct prediction of the current sample in the distillation process. Samples with higher domain expertise weight have higher weight in the distillation process.

![image.png](https://cdn.nlark.com/yuque/0/2022/png/2556002/1647500806787-eb2851dc-8213-40ff-aff7-aa9fc4bd44f1.png#clientId=u0e51cb8b-d6a9-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=298&id=uc221ce22&margin=%5Bobject%20Object%5D&name=image.png&originHeight=474&originWidth=1126&originalType=binary&ratio=1&rotation=0&showTitle=false&size=260756&status=done&style=none&taskId=ufc518b45-cdec-4ad2-929e-65961b1fc08&title=&width=309)


## Requirement

This project uses the lightly version of oneflow. You can use the following command to install.
CPU：
```bash
python3 -m pip install -f  https://staging.oneflow.info/branch/master/cpu  --pre oneflow
```
GPU：
```bash
python3 -m pip install -f  https://staging.oneflow.info/branch/master/cu112  --pre oneflow
```
You can install other dependencies using the following command.
```bash
pip install -r requirements.txt
```
## Download the Pre-training Models "bert-base-uncased" and "bert-tiny-uncased"
```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/knowledge_distillation_metakd/bert-base-uncased-oneflow.tar.gz
tar -xzf bert-base-uncased-oneflow.tar.gz
```
```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/knowledge_distillation_metakd/bert-tiny-uncased-oneflow.tar.gz
tar -xzf bert-tiny-uncased-oneflow.tar.gz
```
## Step1：Environmental Preparation
Download the sample data set and divide it.
```bash
bash 1_get_data.sh
```

## Step2：Preprocessing Sample Datasets

Generate the meta weight required for training and unify the test set format:

```bash
bash 2_process_data.sh
```

## Step3：Training Meta Teacher
During training, you need to specify `use_sample_weight` and `use_domain_loss` as `True` and set the value of `domain_loss_weight`.
```bash
bash 3_train_teacher.sh
```

## Step4：Student Model of Distillation Corresponding Field -- Middle Layer Output of Fitting Teacher Model

You need to specify the save path of the Meta Teacher model: `teacher_model_path`, and set the variable `distill_stage` to `first`. In addition, `checkpoint_dir` of the first stage distillation will be used as the model input `pretrain_model_name_or_path` of the second stage distillation.

```bash
bash 4_student_first.sh
```
## Step5：Student Model of Distillation Corresponding Field -- Distillation Loss Function Training Student Model
In the second stage, it is also necessary to formulate the saving path of the Meta Teacher model and set the variable `distill_stage` to `second`. At the same time, ensure that `pretrain_model_name_or_path` is the model saving location of the first stage.

```bash
bash 5_student_second.sh
```

## Step6：Evaluation Student Model

```bash
bash 6_eval.sh
```
