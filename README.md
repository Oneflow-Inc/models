Please refer to the [official repository](https://github.com/Oneflow-Inc/models) for detailed documentation.

## Running experiments in the OCCL paper
```shell
cd Vision/classification/image/resnet50/examples
bash train_ofccl_graph_distributed_fp32.sh <NUM_LOCAL_GPUS>
```

Notes:
- Prepare the ImageNet dataset in advance.
