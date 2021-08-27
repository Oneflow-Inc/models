import os
import sys
from shutil import copy
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import var

from google.protobuf import text_format
from numpy.lib.index_tricks import _fill_diagonal_dispatcher
from oneflow.core.framework.variable_meta_info_pb2 import VariableMetaInfo


def load_meta_info(path):
    with open(os.path.join(path, 'meta'), 'r') as f:
        meta_info = VariableMetaInfo()
        text_format.Parse(f.read(), meta_info)
    
    if meta_info.data_type in [2, 3]:
        dtype = np.float32
    elif meta_info.data_type in [4, 5, 6, 7]:
        dtype = np.int32
    else:
        assert 'UNIMPLEMENTED'

    return dtype, list(meta_info.shape.dim)


def npy_compare(lhs_path, rhs_path):
    lhs = np.load(lhs_path)
    rhs = np.load(rhs_path)
    #if not np.allclose(lhs, rhs):
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
        for name in filter(lambda f: f.endswith('.npy'), files):
            lhs_path = os.path.join(root, name)
            rhs_path = os.path.join(rhs, os.path.relpath(lhs_path, lhs))
            if os.path.exists(rhs_path) and os.path.isfile(rhs_path):
                if not npy_compare(lhs_path, rhs_path):
                    mean, std, max_, min_ = npy_diff(lhs_path, rhs_path)
                    print(lhs_path, f'mean={mean}, std={std}, max={max_}, min={min_}')
                    diff += 1
                else:
                    same += 1
            else:
                print('{} ignore'.format(lhs_path))
                ignore += 1
    print('same:', same)
    print('diff:', diff)
    print('ignore:', ignore)

import zlib
def crc32(filename):
    with open(filename, 'rb') as f:
        data = f.read()
        print(filename, zlib.crc32(data))
        return zlib.crc32(data)


def file_cksum_compare(lhs_path, rhs_path):
    lhs = crc32(lhs_path)
    rhs = crc32(rhs_path)
    if lhs != rhs:
        print(lhs)
        print(rhs)
    return lhs == rhs


def get_varible_name(var_org):
    # for item in sys._getframe().f_locals.items():
    #     print(item[0],item[1])
    # for item in sys._getframe(1).f_locals.items():
    #     print(item[0],item[1])
    for item in sys._getframe(2).f_locals.items():
        if (var_org is item[1]):
            return item[0]


def dump_to_npy(tensor, root='./output', sub='', name=''):
    if sub != '':
        root = os.path.join(root, str(sub))
    if not os.path.isdir(root):
        os.makedirs(root)

    var_org_name = get_varible_name(tensor) if name == '' else name
    path = os.path.join(root, f'{var_org_name}.npy')
    if not isinstance(tensor, np.ndarray):
        # tensor = tensor.to_local().numpy()
        tensor = tensor.numpy()
    np.save(path, tensor)


def save_param_npy(module, root='./output'):
    for name, param in module.named_parameters():
        # if name.endswith('bias'):
        dump_to_npy(param.numpy(), root=root, sub=0, name=name)


def param_hist(param, name, root='output'):
    print(name, param.shape)
    # print(param.flatten())

    # the histogram of the data
    n, bins, patches = plt.hist(param.flatten(), density=False, facecolor='g')

    # plt.xlabel('Smarts')
    # plt.ylabel('value')
    plt.title(f'Histogram of {name}')
    # plt.xlim(40, 160)
    # plt.ylim(0, 0.03)
    plt.grid(True)
    plt.savefig(os.path.join(root, f"{name}.png"))
    plt.close()


def save_param_hist_pngs(module, root='output'):
    for name, param in module.named_parameters():
        # if name.endswith('bias'):
        param_hist(param.numpy(), name, root=root)



def walk_compare(lhs, rhs, func, endswith='out', detail_func=None):
    assert os.path.isdir(lhs)
    assert os.path.isdir(rhs)

    same = 0
    diff = 0
    ignore = 0
    for root, dirs, files in os.walk(lhs):
        for name in filter(lambda f: f.endswith(endswith), files):
            lhs_path = os.path.join(root, name)
            rhs_path = os.path.join(rhs, os.path.relpath(lhs_path, lhs))
            if os.path.exists(rhs_path) and os.path.isfile(rhs_path):
                if not func(lhs_path, rhs_path):
                    print('{} False'.format(lhs_path))
                    if detail_func:
                        detail_func(lhs_path, rhs_path)
                    diff += 1
                else:
                    same += 1
            else:
                print('{} ignore'.format(lhs_path))
                ignore += 1
    print('same:', same)
    print('diff:', diff)
    print('ignore:', ignore)


def get_meta_info_from_out_file(out_file):
    # path = os.path.join(os.path.dirname(out_file), 'meta')
    path = os.path.dirname(out_file)
    if os.path.isfile(path + '/meta'):
        return load_meta_info(path)
    else:
        return None


def var_to_ndarray_compare(lhs_path, rhs_path):
    meta_l = get_meta_info_from_out_file(lhs_path)
    meta_r = get_meta_info_from_out_file(rhs_path)

    if meta_l != meta_r and (meta_l is not None) and (meta_r is not None):
        print('meta different', meta_l, meta_r)
        return False
    
    meta = meta_l if meta_l is not None else meta_r
    
    def file_to_npy(path):
        return np.fromfile(path, dtype=meta[0]).reshape(meta[1])
    
    lhs = file_to_npy(lhs_path)
    rhs = file_to_npy(rhs_path)
    diff = np.absolute(lhs - rhs)
    div = lhs / rhs
    var_name = os.path.basename(os.path.dirname(lhs_path))
    print(var_name, diff.mean(), diff.std(), diff.max(), div.mean(), div.std(), div.max())

    # if 'running_mean'
    return True


def var_hist(out_file):
    root = os.path.dirname(os.path.dirname(out_file))
    var_name = os.path.basename(os.path.dirname(out_file))
    if 'System-Train-TrainStep' in var_name:
        return
    fig_file = os.path.join(root, f'{var_name}.png')
    print(fig_file)

    meta = get_meta_info_from_out_file(out_file)
    param = np.fromfile(out_file, dtype=meta[0])
    # the histogram of the data
    n, bins, patches = plt.hist(param, density=False, facecolor='g')

    plt.title(f'Histogram of {var_name}')
    plt.grid(True)
    plt.savefig(fig_file)
    plt.close()


def walk_and_do(path, func, endswith='out'):
    assert os.path.isdir(path)

    for root, dirs, files in os.walk(path):
        for name in filter(lambda f: f.endswith(endswith), files):
            filename = os.path.join(root, name)
            func(filename)


if __name__ == '__main__':
    # compare ckpt values of two folder
    # path0 = '/ssd/xiexuan/models/resnet50/iter0_ckpt'
    # path1 = '/ssd/xiexuan/OneFlow-Benchmark/Classification/cnns/output/snapshots/model_save-20210826105011/snapshot_epoch_0_iter0'
    # walk_compare(path0, path1, var_to_ndarray_compare, 'out')

    # gen ckpt weight hist figures
    path = '/ssd/xiexuan/OneFlow-Benchmark/Classification/cnns/init_ckpt_by_lazy'
    path = '/ssd/xiexuan/models/resnet50/init_ckpt_by_graph'
    walk_and_do(path, var_hist)
