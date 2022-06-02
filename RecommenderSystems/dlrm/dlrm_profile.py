import os
import sys

test_name = "dlrm_profile"
nsys = '/usr/local/cuda-11.6/bin/nsys profile --stats=true '
#nsys = '/usr/local/cuda-11.5/bin/nsys profile --stats=true '

data_dir = "/data/criteo1t/criteo1t_dlrm_parquet"
persistent_path = './persistent'
script_path = 'dlrm_train_eval.py'
#script_path = 'dlrm_prefetch_train.py'

env = ''
#env += "NCCL_DEBUG=INFO "
#env += "ONEFLOW_DEBUG_MODE=INFO "
env += "ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1 "
env += "ONEFLOW_EAGER_LOCAL_TO_GLOBAL_BALANCED_OVERRIDE=1 "
env += "ONEFLOW_ONE_EMBEDDING_ENABLE_QUANTIZED_COMM=1 "

dl = sys.executable + " -m oneflow.distributed.launch "
dl += "--nproc_per_node 4 "
dl += "--nnodes 1 "
dl += "--node_rank 0 "
dl += "--master_addr 127.0.0.1 "
dl += f"{script_path} "

cfg = ""
cfg += "--train_batches 300 "
cfg += "--eval_interval 0 "
cfg += f"--persistent_path {persistent_path} "
cfg += f"--data_dir {data_dir} "
cfg += "--store_type device_mem " 
cfg += "--amp " 


cmd = dl + cfg 
cmd = nsys + f"-o {test_name} " + dl + cfg 
os.system(f'rm -rf {persistent_path}/*')
os.system("echo " + env + cmd + f" | tee {test_name}.log")
os.system(env + cmd + f" | tee {test_name}.log")

