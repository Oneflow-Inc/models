## Knowledge Distillation
Oneflow implement for（Knowledge Distillation, KD）Algorithm

---
## Knowledge Distillation：
In order to avoid the problem that the super-large model is not conducive to online, knowledge distillation aims to solve how to use a model with a small parameter amount to make it have a comparable effect with the large model.

KD mainly consists of two parts, namely Teacher and Student:
- Teacher：denotes the original large model, usually learning directly on supervised data. In the inference stage, obtain the probability distribution of each sample;
- Student：denotes the small model to be obtained, which learns based on the probability distribution of each sample obtained by the Teacher model, that is, learns the prior of the Teacher model

Therefore, the simple KD is mainly divided into two steps, first train the Teacher model, and then train the Student model.

## Data Acqusition
MNIST（training data:60000，testing data:10000）
Oneflow has implemented the data acquisition code (ofrecord format), so the subsequent code will be downloaded automatically without manual downloading. If you view specific data, please look up the details:http://yann.lecun.com/exdb/mnist/


## Experiment Settings

#### Step1：Training the Teacher Model：

```shell
python3 main.py \
	--model_type teacher \
	--epoch 10 \
	--temperature 5
```
Running effect: The best accuracy rate on the test set is 98.98%, and the model is saved to the output:
![Teacher](images/kd_teacher.png)

#### Step2：Training the Student Model：
Select the best Teacher model on the test set, and then use the soft label obtained by the Teacher model as supervision to train the Student model
Obtain the model file with the highest accuracy（e.g. `output/model_save-2021-06-20-09:18:50`），then:
```shell
python3 main.py \
	--model_type student \
	--load_teacher_from_checkpoint \
	--load_teacher_checkpoint_dir ./output/model_save-2021-06-20-09:18:50 \
	--epoch 50 \
	--temperature 5
```
We can obtain 89.19%。
![Student](images/kd_student.png)

