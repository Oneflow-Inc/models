# GLM-OneFlow

## Install:
install [oneflow](https://github.com/Oneflow-Inc/oneflow)
```shell
  pip3 install -r requirements.txt
```

## Download datasets 
dir `other` should be in models/NLP/GLM/
```shell
  wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/NLP/other.zip
  unzip other.zip
```

## Download initial model
```shell
  wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/mo.pt
```

## Run eager training

### Single GPU
you can choose GPU ID by setting `CUDA_VISIBLE_DEVICES=gpu_id`
```shell
CUDA_VISIBLE_DEVICES=0 bash pretrain.sh
```


### Multi Machine - Multi GPU
you can set `pretain_ddp.sh` for training
```shell
bash pretrain_ddp.sh
```

**Single Machine - Multi GPU**
```shell
_DEVICE_NUM_PER_NODE=4  #set number of gpus
_MASTER_ADDR=127.0.0.1
_NUM_NODES=1
_NODE_RANK=0
_MASTER_PORT=8089 # set to unused port
```

**Multi Machine - Multi GPU**
>NOTE: All machines must access each other

For example, if you want to run scripts in Machine A and Machine B. Machine A is the master Machine 

for Machine A(IP Address 192.168.0.1), set `pretrain_ddp.sh`
```shell
_DEVICE_NUM_PER_NODE=4     #set number of gpus in machine A
_MASTER_ADDR=192.168.0.1   #master machine's ip address
_NUM_NODES=2               #total number of machines
_NODE_RANK=0               #this machine's id
_MASTER_PORT=8089          #set to master machine's unused port
```

for Machine B(IP Address 192.168.0.2), set `pretrain_ddp.sh`
```shell
_DEVICE_NUM_PER_NODE=4     #set number of gpus in machine B
_MASTER_ADDR=192.168.0.1   #master machine's ip address
_NUM_NODES=2               #total number of machines
_NODE_RANK=1               #this machine's id
_MASTER_PORT=8089          #set to master machine's unused port
```

run `bash pretrain_ddp.sh` in machine A and machine B

## Run graph training
you can set `pretain_graph.sh` for training
```shell
bash pretrain_graph.sh
```

**Single GPU**

run `bash pretrain_graph.sh`
Same as eager 
```shell
_DEVICE_NUM_PER_NODE=1
_MASTER_ADDR=127.0.0.1
_NUM_NODES=1
_NODE_RANK=0
```

**Multi machine - Multi GPU**

coming soon