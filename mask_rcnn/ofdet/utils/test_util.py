import numpy as np
import os
from oneflow.python.framework.local_blob import LocalBlob


def GetSavePath():
    return "./saved_blobs/"


def Save(name):
    path = GetSavePath()
    if not os.path.isdir(path):
        os.makedirs(path)

    def _save(x):
        if isinstance(x, LocalBlob):
            x = x.numpy(0)
        else:
            assert False
        np.save(os.path.join(path, name + "_" + str(x.shape)), x)

    return _save
