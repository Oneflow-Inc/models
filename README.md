Please refer to the [official repository](https://github.com/Oneflow-Inc/models) for detailed documentation.

## Running experiments in the OCCL paper
```shell
cd Vision/classification/image/resnet50/examples
bash train_ofccl_graph_distributed_fp32.sh <NUM_LOCAL_GPUS>
```

Notes:
- Prepare the ImageNet dataset in advance.
- If the environment virable `ONEFLOW_ENABLE_OFCCL` in [train_ofccl_graph_distributed_fp32.sh](https://github.com/Panlichen/models/blob/test_ofccl/Vision/classification/image/resnet50/examples/train_ofccl_graph_distributed_fp32.sh#L16) is set to `1`, OCCL will be used during training; otherwise, NCCL will be employed.
