import os
import sys
import argparse
import datetime
from dateutil import tz

num_gpus = 4
persistent_path = './persistent'
table_size_array = [39884406, 39043, 17289, 7420, 20263, 3, 7120, 1543, 63, 38532951, 2953546, 403346, 10, 2208, 11938, 155, 4, 976, 14, 39979771, 25641295, 39664984, 585935, 12972, 108, 36]
table_size_array = ','.join([str(i) for i in table_size_array])
num_eval_examples = 89137319
eval_batch_size = 55296 
eval_batchs= num_eval_examples // eval_batch_size
warmup_batches = 2500
decay_batches = 15406
train_batch_size = num_gpus * 6912
#train_batch_size = 69120
num_train_samples = 4195197692
train_batches = num_train_samples // train_batch_size + 1
decay_start = train_batches - decay_batches + 3700

env = ""
env += "ONEFLOW_ONE_EMBEDDING_ENABLE_QUANTIZED_COMM=0 "
env += "ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1 "
env += "ONEFLOW_EAGER_LOCAL_TO_GLOBAL_BALANCED_OVERRIDE=1 "
env += "ONEFLOW_FUSE_MODEL_UPDATE_CAST=1 "
env += "ONEFLOW_ENABLE_MULTI_TENSOR_MODEL_UPDATE=1 "
env += "ONEFLOW_ONE_EMBEDDING_USE_SYSTEM_GATHER=0 "

cfg = ""
cfg += "--eval_interval 100000 "
cfg += "--model_save_dir ckpt "
cfg += "--one_embedding_key_type int32 "
cfg += f"--data_dir /RAID0/xiexuan/dlrm_parquet_int32 "
cfg += f"--persistent_path {persistent_path} "
cfg += f"--store_type device_mem "
cfg += f"--table_size_array {table_size_array} "
cfg += f"--train_batch_size {train_batch_size} "
#cfg += f"--train_batches {train_batches} "
cfg += f"--train_batches 10000 "
cfg += f"--eval_batches {eval_batchs} "
cfg += f"--eval_batch_size {eval_batch_size} "
cfg += f"--warmup_batches {warmup_batches} "
cfg += f"--decay_start {decay_start} "
cfg += f"--decay_batches {decay_batches} "
cfg += f"--amp "


dl = sys.executable + " -m oneflow.distributed.launch "
dl += f"--nproc_per_node {num_gpus} "
dl += "--nnodes 1 "
dl += "--node_rank 0 "
dl += "--master_addr 127.0.0.1 "
dl += "dlrm_prefetch_train.py "


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="flags for OneFlow DLRM")
    parser.add_argument("--log_path", type=str, default="commits.log")
    args = parser.parse_args()
    ext_envs = [
        "ONEFLOW_GRAPH_PLACE_TRAINING_STATE_ON_ALL_RANKS",
        "ONEFLOW_ONE_EMBEDDING_EMBEDDING_GRADIENT_SHUFFLE_INDEPENTENT_STREAM",
        "ONEFLOW_ONE_EMBEDDING_EMBEDDING_SHUFFLE_INDEPENTENT_STREAM",
        "ONEFLOW_ONE_EMBEDDING_FUSE_EMBEDDING_INTERACTION",
        "ONEFLOW_ONE_EMBEDDING_FUSED_MLP_ASYNC_GRAD",
        "ONEFLOW_ONE_EMBEDDING_FUSED_MLP_GRAD_OVERLAP_ALLREDUCE",
        "ONEFLOW_ONE_EMBEDDING_FUSED_MLP_GRAD_UNABLE_ALLREDUCE",
    ]
    for i in range(10):
        # test baseline
        cmd = env + dl + cfg 
        os.system(f'rm -rf {persistent_path}*')
        os.system(f'echo {cmd}')
        os.system(cmd + f" | tee baseline_{i}.log")

        # test split allreduce
        cmd = env + dl + cfg + "--split_allreduce " 
        os.system(f'rm -rf {persistent_path}*')
        os.system(f'echo {cmd}')
        os.system(cmd + f" | tee split_allreduce_{i}.log")

        # test envs
        for ext_env in ext_envs:
            test_name = ext_env
            cmd = env + ext_env + "=1 " + dl + cfg 
            os.system(f'rm -rf {persistent_path}*')
            os.system(f'echo {cmd}')
            os.system(cmd + f" | tee {test_name}_{i}.log")


