# Resnet50

Training Resnet50 on Imagenet Dataset using [OneFlow](https://github.com/Oneflow-Inc/oneflow#install-with-pip-package).

## Quick Start
### 1. Prepare Traning Data 

For quick start, you can download the [mini-imagenet](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/dataset/imagenet/mini-imagenet.zip) to train resnet50 on single device. If you want to train resnet50 with the complete imagenet dataset, please see the following instructions.

### 2. Train on single device

Oneflow supports execution in eager mode or graph mode.


#### Eager Execution
```bash
bash examples/train_eager.sh
```

#### Graph Execution

```bash
bash examples/train_graph.sh
```

### 3.  Distributed Training

#### Prepare Ofrecord for The Full Imagenet Dataset
please refer to: https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/master/Classification/cnns/tools.


Oneflow launches the distributed training the same way as Pytorch. For more details, please refer to the docs: https://oneflow.readthedocs.io/en/master/distributed.html?highlight=distributed#oneflow-distributed

But for know, we have only tested the distributed training on single node with 8 devices. 

#### Train FP32 Model with Distributed Data Prallel

```
bash examples/train_ddp_fp32.sh
```

#### Graph Training for FP32

```
bash examples/train_graph_distributed_fp32.sh
```

#### Graph Training for AMP (auto-mixed precision)

```
bash examples/train_graph_distributed_fp16.sh
```

#### explaination of command parameters for `train.py` script

```bash
--channels-last					 	
									Use the NHWC dataformat.
--ddp								
									Train resnet50 with eager distributed data parallel.
--graph								
									Train resnet50 with graph mode.
--use-fp16						 	
									Whether to enable amp training.
--use-gpu-decode                    
									Use gpu to decode the data packed in ofrecord, only supported in graph mode. 
--scale-grad					 	
									Whether to scale gradient when training in fp32 with graph mode. 
--skip-eval						 	
									Whether to skip the valution and the end of the traning epoch.
--zero-init-residual			  	
									Whether to zero-initialize the last BN in each residual branch.
--save-init						 	
									Whether to save the parameters right after the initialization.
--print-timestamp					
									Whether to print the time stamp.
--synthetic-data					
									To use the synthetic data, only for testing the throughput, no need to provide the real data.

--label-smoothing LABEL_SMOOTHING
									label smoothing rate.
--learning-rate LEARNING_RATE
									learning rate of sgd.
--load-path LOAD_PATH
									load the model provided by user after model initialization.
--momentum MOMENTUM
									SGD momentum.					
--num-devices-per-node NUM_DEVICES_PER_NODE
									The device number, must be the same for each node.
--num-epochs NUM_EPOCHS
									How many epoches to train the resnet50.
--num-nodes NUM_NODES
									The number of nodes to launch the distributed training.
--ofrecord-part-num OFRECORD_PART_NUM
									ofrecord part numbers.
--ofrecord-path OFRECORD_PATH
									The ofrecord path of imagenet dataset.
--samples-per-epoch
									The total number of training samples.
--val-samples-per-epoch
									The total number of validation samples.
--print-interval PRINT_INTERVAL
									The intervals of iters to print the loss during tranining.
--save-path SAVE_PATH
									Path to save the checkpoints during training.
--train-batch-size TRAIN_BATCH_SIZE
									Train batch size of each device.								
--val-batch-size VAL_BATCH_SIZE
									Val batch size of each device.		
--warmup-epochs WARMUP_EPOCHS
									The number of epoches to do the warmup learning rate scheduling.	
```

### 4. Inference on Single Image

#### Download Pretrained Models for Imagenet

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/classification/resnet50_imagenet_pretrain_model.tar.gz
tar zxf resnet50_imagenet_pretrain_model.tar.gz
```

#### Inference with Eager Execution
```bash
bash examples/infer.sh
```

#### Inference with Graph Execution
```bash
bash examples/infer_graph.sh
```

## Util
### 1. Model Compare
Compare Resnet50 model on different training mode (Graph / Eager)
```bash
bash check/check.sh
```
Compare results will be saved to `./results/check_report.txt`
Compare info txt will be saved to `./results/default`

Compare Results Picture
```bash
bash check/draw.sh
```
The pictures will be saved to `./results/pictures`
