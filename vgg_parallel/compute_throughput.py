import sys

import numpy as np

with open(sys.argv[1], "r") as f:
    res = [float(i) for i in f.readlines()]

drop_num = int(len(res) * 0.1)
res = sorted(res)
print(f"average throughput is {np.mean(res[drop_num:-drop_num])}")
