import os
import sys
from shutil import copy
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import var


def npy_compare(lhs_path, rhs_path):
    lhs = np.load(lhs_path)
    rhs = np.load(rhs_path)
    # if not np.allclose(lhs, rhs):
    #    print(lhs)
    #    print(rhs)
    return np.allclose(lhs, rhs)


def npy_diff(lhs_path, rhs_path):
    lhs = np.load(lhs_path)
    rhs = np.load(rhs_path)
    diff = np.absolute(lhs - rhs)
    return diff.mean(), diff.std(), diff.max(), diff.min()


def walk_compare_npy(lhs, rhs):
    assert os.path.isdir(lhs)
    assert os.path.isdir(rhs)

    same = 0
    diff = 0
    ignore = 0
    for root, dirs, files in os.walk(lhs):
        for name in filter(lambda f: f.endswith(".npy"), files):
            lhs_path = os.path.join(root, name)
            rhs_path = os.path.join(rhs, os.path.relpath(lhs_path, lhs))
            if os.path.exists(rhs_path) and os.path.isfile(rhs_path):
                if not npy_compare(lhs_path, rhs_path):
                    mean, std, max_, min_ = npy_diff(lhs_path, rhs_path)
                    print(lhs_path, f"mean={mean}, std={std}, max={max_}, min={min_}")
                    diff += 1
                else:
                    same += 1
            else:
                print("{} ignore".format(lhs_path))
                ignore += 1
    print("same:", same)
    print("diff:", diff)
    print("ignore:", ignore)


import zlib


def crc32(filename):
    with open(filename, "rb") as f:
        data = f.read()
        print(filename, zlib.crc32(data))
        return zlib.crc32(data)


def var_compare(lhs_path, rhs_path):
    lhs = crc32(lhs_path)
    rhs = crc32(rhs_path)
    if lhs != rhs:
        print(lhs)
        print(rhs)
    return lhs == rhs


def walk_compare_of_variable(lhs, rhs):
    assert os.path.isdir(lhs)
    assert os.path.isdir(rhs)

    same = 0
    diff = 0
    ignore = 0
    for root, dirs, files in os.walk(lhs):
        for name in filter(lambda f: f.endswith("out"), files):
            lhs_path = os.path.join(root, name)
            rhs_path = os.path.join(rhs, os.path.relpath(lhs_path, lhs))
            if os.path.exists(rhs_path) and os.path.isfile(rhs_path):
                if not var_compare(lhs_path, rhs_path):
                    print("{} False".format(lhs_path))
                    diff += 1
                else:
                    same += 1
            else:
                print("{} ignore".format(lhs_path))
                ignore += 1
    print("same:", same)
    print("diff:", diff)
    print("ignore:", ignore)


def get_varible_name(var_org):
    # for item in sys._getframe().f_locals.items():
    #     print(item[0],item[1])
    # for item in sys._getframe(1).f_locals.items():
    #     print(item[0],item[1])
    for item in sys._getframe(2).f_locals.items():
        if var_org is item[1]:
            return item[0]


def dump_to_npy(tensor, root="./output", sub="", name=""):
    if sub != "":
        root = os.path.join(root, str(sub))
    if not os.path.isdir(root):
        os.makedirs(root)

    var_org_name = get_varible_name(tensor) if name == "" else name
    path = os.path.join(root, f"{var_org_name}.npy")
    if not isinstance(tensor, np.ndarray):
        tensor = tensor.to_local().numpy()
    np.save(path, tensor)


def save_param_npy(module, root="./output"):
    for name, param in module.named_parameters():
        # if name.endswith('bias'):
        dump_to_npy(param.numpy(), root=root, sub=0, name=name)


def param_hist(param, name, root="output"):
    print(name, param.shape)
    # print(param.flatten())

    # the histogram of the data
    n, bins, patches = plt.hist(param.flatten(), density=False, facecolor="g")

    # plt.xlabel('Smarts')
    # plt.ylabel('value')
    plt.title(f"Histogram of {name}")
    # plt.xlim(40, 160)
    # plt.ylim(0, 0.03)
    plt.grid(True)
    plt.savefig(os.path.join(root, f"{name}.png"))
    plt.close()


def save_param_hist_pngs(module, root="output"):
    for name, param in module.named_parameters():
        # if name.endswith('bias'):
        param_hist(param.numpy(), name, root=root)


if __name__ == "__main__":
    # walk_compare_of_variable('init_ckpt', '/ssd/xiexuan/OneFlow-Benchmark/Classification/cnns/loaded_init_ckpt')
    walk_compare_npy(
        "output", "/ssd/xiexuan/OneFlow-Benchmark/Classification/cnns/output"
    )
