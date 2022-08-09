# Knowledge Distillation
[OneFlow](https://github.com/Oneflow-Inc/oneflow#install-with-pip-package) implement for（Knowledge Distillation, KD）Algorithm

---
## Knowledge Distillation：
In order to avoid the problem that the super-large model is not conducive to online, knowledge distillation aims to solve how to use a model with a small parameter amount to make it have a comparable effect with the large model.

KD mainly consists of two parts, namely Teacher and Student:
- Teacher: denotes the original large model, usually learning directly on supervised data. In the inference stage, obtain the probability distribution of each sample;
- Student: denotes the small model to be obtained, which learns based on the probability distribution of each sample obtained by the Teacher model, that is, learns the prior of the Teacher model

Therefore, the simple KD is mainly divided into two steps. First train the Teacher model, and then train the Student model.

In addition, we will also test the student model that has not been taught by the Teacher model for comparison to ensure that the algorithm is indeed effective.

## Data Acqusition
MNIST（training data:60000，testing data:10000）
Oneflow has implemented the data acquisition code (ofrecord format), so the subsequent code will be downloaded automatically without manual downloading. If you view specific data, please look up the details: http://yann.lecun.com/exdb/mnist/

## requirement

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

## Experiment Settings

You can use the command below to perform the training process. Modify the parameter `--model_type` for different training tasks. The optional parameters are explained as follows:
```bash
bash train.sh
```

"teacher": training Teacher model.
"student_kd": train the student model under the guidance of the Teacher model. 
"student": train the student model (without the guidance of the Teacher model).
"compare": compare the advantages of the above three models.

### Step1：Training the Teacher Model：
Set the parameter `--model_type` to `teacher` and run the command to train the Teacher model.

We can obtain 98.19%。

### Step2：Training the Student Model with Teacher Model：
Set the parameter `--model_type` to `student_kd` and run the command to train the student_kd model.

Select the best Teacher model on the test set, and then use the soft label obtained by the Teacher model as supervision to train the student model.

Obtain the model file with the highest accuracy（e.g. `output/model_save/teacher`），then modify the parameter `--load_teacher_checkpoint_dir="./output/model_save/teacher"`to load the Teacher model, then the student model is trained under the guidance of the Teacher model.

We can obtain 89.19%。

### Step3：Training the Student Model without Teacher Model：
Set the parameter `--model_type` to `student` and run the command to train the student model.

We can obtain 89.19%。

### Step4：Compare the Performance of the Three Models：

Set the parameter `--model_type` to `compare` to run the above three tasks at one time, and verify that the knowledge distillation algorithm does improve the performance of the student model.
![](./output/images/compare_result.jpg)

### Explaination of Command Parameters for `train.py` Script
| parameters     | meaning     | remarks     |
| -------- | -------- | -------- |
| --model_save_dir | Path to save the model. | The model names are: teacher，student_kd, student. |
| --image_save_name | Save name of pictures after training. |  |
| --load_teacher_checkpoint_dir | Path to load Teacher model. |  |
| --model_type | Type of model. | There are only four options, teacher, student_ kd, student and compare. |
| --epochs | How many epoches to train the models. |  |
| --batch_size | Train batch size of each device. |  |
| --temperature | Temperature of distillation Teacher model. |  |
| --alpha | Weight coefficient when calculating student model loss. |  |


## Inference on Single Image

### Quick Start
You can run the following command to infer a single picture:
```bash
bash infer.sh
```

### Explaination of Command Parameters for `infer.py` Script
You can also modify the parameters to predict according to the meaning of the parameters.

| parameters     | meaning     | remarks     |
| -------- | -------- | -------- |
| --model_type | Type of model. | There are only three options:  teacher, student_ kd and student. |
| --model_load_dir | Path from which to load the model. |  |
| --image_save_name | Save the predicted picture to this path. |  |
| --picture_index | Use flowvision to load the data set and select which picture for infering. | The range is 0-9999 |
| --temperature | Temperature of distillation the model. |  |

The predicted results are similar to the following figure:
![](./output/images/infer_result.jpg)
From top to bottom are the figure, the predicted result, and the result after distillation.
