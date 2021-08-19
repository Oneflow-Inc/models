import os
import numpy as np
from util import load_meta_info, var_to_npy

eager_root = '/home/xiexuan/sandbox/models/wide_and_deep/output'
ckpt_path_eager = os.path.join(
    eager_root, "iter0_checkpoint/deep_embedding.weight")

lazy_root = "/home/xiexuan/sandbox/OneFlow-Benchmark/ClickThroughRate/WideDeepLearning"
ckpt_path_lazy = os.path.join(lazy_root, "iter0_checkpoints/deep_embedding")
ckpt_path_init = os.path.join(lazy_root, 'baseline_checkpoint/deep_embedding')

dtype, shape = load_meta_info(ckpt_path_eager)
w = var_to_npy(ckpt_path_init, dtype, shape)

e = var_to_npy(ckpt_path_eager, dtype, shape)
l = var_to_npy(ckpt_path_lazy, dtype, shape)

eager_diff = np.load(os.path.join(eager_root, "0/deep_weight_grad.npy"))
lazy_diff = np.load(os.path.join(lazy_root, "deep_embedding_weight_diff.npy"))

is_diff_close = np.allclose(eager_diff, lazy_diff)
print('is diff close', is_diff_close)

lr = 0.001

print('is updated weight close', np.allclose(e, l))
print(e)
print(l)
# print((w - l)/lr)
# print(lazy_diff)
# print(lazy_diff/(w - l)*lr)
# print(eager_diff/(w - e)*lr)
