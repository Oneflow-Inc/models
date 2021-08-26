
# DLA

## Train on [DLAdataset_ofrecord](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/DLA/DLAdataset_ofrecord.zip) Dataset

### Prepare Traning Data And Pretrain Models

#### Download Ofrecord

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/DLA/DLAdataset_ofrecord.zip
unzip DLAdataset_ofrecord.zip
```

### Download Pretrain Models

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/DLA/pretrain_model.zip
unzip pretrain_model.zip
```

### Run Oneflow Training script
We have installed visdom to visualize the training model, and run the following program to enter http://localhost:8097/ get the training curve.
```
pip3 install -r requirements.txt --user
python -m visdom.server
bash train_oneflow.sh
```


## Inference on Single Image

```bash
bash infer.sh
```


### Performer of model
|               |       val(Top1)     |
|    :-----:    | :-----------------: |
| DLA-pytorch   |        0.912        |
| DLA-oneflow   |        0.922        |

