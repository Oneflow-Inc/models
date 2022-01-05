import os
import yaml
from yacs.config import CfgNode as CN


_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU
_C.DATA.BATCH_SIZE = 128
_C.DATA.DATA_PATH = ''
_C.DATA.DATASET = 'imagenet'
# _C.DATA.IMAGE_SIZE = 1024


# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.name = ''
_C.TRAIN.iter = 50000
_C.TRAIN.start_iter = 0
_C.TRAIN.nlr = 0.0002
_C.TRAIN.nbeta1 = 0.5
_C.TRAIN.nbeta2 = 0.999
_C.TRAIN.save = ''
_C.TRAIN.load = ''
_C.TRAIN.dataloader_workers = 8
_C.TRAIN.save_interval = 100
_C.TRAIN.im_size = 1024
_C.TRAIN.cpkt = ''
_C.TRAIN.multi_gpu = ''

# -----------------------------------------------------------------------------
# Generator settings
# -----------------------------------------------------------------------------
_C.G = CN()
_C.G.ngf = 64
_C.G.nz = 256


# -----------------------------------------------------------------------------
# Generator settings
# -----------------------------------------------------------------------------
_C.D = CN()
_C.D.ndf = 64


def get_config():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    return config