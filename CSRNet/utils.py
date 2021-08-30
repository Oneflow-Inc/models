import h5py
import oneflow as flow
import shutil
import numpy as np
import os

def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())
def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = flow.Tensor(np.asarray(h5f[k]))
            v.copy_(param)
def save_checkpoint(state, is_best,task_id, filename='checkpoints/'):

    del_file(filename+str(int(task_id) - 1))
    flow.save(state['state_dict'], filename+task_id)
    if is_best:
        file_path= 'checkpoints/model_best'
        del_file(file_path)
        shutil.copytree(filename+task_id, file_path)


def del_file(filepath):
    """
    Delete all files or folders in a directory
    :param filepath:
    :return:
    """

    if os.path.exists(filepath):
        del_list = os.listdir(filepath)
        for f in del_list:
            file_path = os.path.join(filepath, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        shutil.rmtree(filepath)
