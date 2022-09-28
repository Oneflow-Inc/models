# 2022 BJTU Summer Championship Advanced Track First Place

## Preparation Stage

In order to run the project, first you need to download the dataset

Dataset URL https://www.datafountain.cn/competitions/569/datasets

There are three files in total, all of which need to be downloaded

**training_dataset**

**test_dataset**

**submit_example**

After downloading the dataset, you need to install the code runtime environment

```cmd
pip3 install -r requirements.txt
```

## Run Code

### 1.run Prepare.py

```python
python3 Prepare.py --train_json train.json --split_p 0.85
```

### 2.run Train.py

```python
python3 Train.py --model_layer 121 --dataset_method 1 --image_train_json train_list.json --image_val_json val_list.json --label_soft True --device cuda --train_batch_size 20 --lr 0.0003 --epochs 50 --standard_acc 0.97
```

### 3.run Predict.py

```python
python3 Predict.py --model_layer 121 --pth_path <model weights path> --image_test_json submit_example.json --device cuda
```
