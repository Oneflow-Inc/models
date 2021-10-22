## install:
>Please first install PyTorch and apex, and then install other dependencies by pip install -r requirements.txt


## 四个目录:
```shell
     GLM_copa:               GLM 的 torch版本
     GLM_copa_oneflow:       GLM 的 oneflow eager单卡
     GLM_copa_torch  :       GLM_copa 去掉分布式的代码，由GLM_copa_oneflow转
     GLM_copa_oneflow_dis:   GLM 的 oneflow graph单卡
```

## 数据集:
>[other.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/NLP/other.zip)解压到四个目录下面


## 注意和优化:
```shell

    没有进行梯度裁剪 : mpu/grad.py

    graph单卡训练没有支持学习率调整方式 : 

    loss计算还不支持autograd.function : mpu/cross_entropy.py
    
    transformer mask 可以写成一个op
```


## 训练 : 
```shell
    训练一个初始化模型到当前目录：
        cd GLM_copa
        打开pretrain_glm.py:train():
               #if args.iteration==100:
               #    torch.save(model.state_dict(),"../mo.pt")
               #    exit(0)
        bash pretrain.sh
    
    训练glm模型:
        bash pretrain.sh
    
    推理copa任务:
        bash infer_copa.sh
```                          