import sys
import glob
import numpy as np
import os


def compute_throughput(file_path):
    with open(file_path, "r") as f:
        res = [float(i) for i in f.readlines()]

    drop_num = int(len(res) * 0.1)
    res = sorted(res)
    return np.mean(res[drop_num:-drop_num])


get_train_cfg = lambda file_path: file_path.split("/")[-1][:-4]

if __name__ == "__main__":
    proc_dir = sys.argv[1]
    all_files = glob.glob(os.path.join(proc_dir, "*.txt"))
    all_res = []
    for file_path in all_files:
        train_cfg = get_train_cfg(file_path)
        throughput = compute_throughput(file_path)
        all_res.append((train_cfg, throughput))

    all_res.sort(key=lambda x: x[-1], reverse=True)
    save_path = sys.argv[2]
    with open(save_path, "w") as f:
        for res in all_res:
            f.write(res[0] + ": " + str(res[1]) + '\n')
