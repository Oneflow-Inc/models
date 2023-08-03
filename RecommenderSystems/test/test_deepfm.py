import os
import sys
import json
import argparse


def run_command():
    pass

def prepare_args(kwargs):
    str_args = ""
    for k, v in kwargs.items():
        str_args += f"--{k}={v} "
    return str_args


if __name__ == "__main__":
    persistent_path = "persistent"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--script_path", type=str, required=True)
    args = parser.parse_args()

    meta_file = os.path.join(args.data_dir, "meta.json")
    assert os.path.isfile(meta_file)
    with open(meta_file, "r") as fp:
        kwargs = json.load(fp)
    del kwargs["field_dtypes"]
    kwargs["data_dir"] = args.data_dir
    kwargs["table_size_array"] = ",".join([str(s) for s in  kwargs["table_size_array"]])
    kwargs["persistent_path"] = persistent_path
    kwargs["store_type"] = "device_mem"
    kwargs["embedding_vec_size"] = 10 

    str_args = prepare_args(kwargs)

    script_path = "/data/xiexuan/git-repos/models/RecommenderSystems/deepfm/deepfm_train_eval.py"
    
    dl = sys.executable + " -m oneflow.distributed.launch "
    dl += "--nproc_per_node 4 "
    dl += "--nnodes 1 "
    dl += "--node_rank 0 "
    dl += "--master_addr 127.0.0.1 "
    dl += f"{args.script_path} "

    cmd = dl + str_args
    print(cmd)
    os.system(f'rm -rf {persistent_path}/*')
    os.system(cmd + f" | tee t.log")

