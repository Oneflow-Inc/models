# Auto-anchor utils

import numpy as np
import oneflow.experimental as flow
import yaml
from tqdm import tqdm

from utils.general import colorstr


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv3 Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)
