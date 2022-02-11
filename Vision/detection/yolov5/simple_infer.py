import argparse
import os
import sys
from pathlib import Path

import cv2
import oneflow as flow
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.flow_utils import time_sync


model = DetectMultiBackend(weights=ROOT / 'yolov5_ckpt', device='cuda:0', dnn=False, data=ROOT / 'data/coco128.yaml')
model = model.eval()

np.random.seed(0)
im = flow.tensor(np.random.rand(4, 3, 640, 640)).to('cuda:0')
im = im.float()
with flow.no_grad()
    # 预热
    for i in range(5):
        pred = model(im, augment=False, visualize=False)
        out = pred.numpy()
    t1 = time_sync()
    # 埋点
    flow._oneflow_internal.profiler.RangePush("oneflow-yolov5-infer")
    for i in range(20):
        flow._oneflow_internal.profiler.RangePush("forward")
        pred = model(im, augment=False, visualize=False)
        flow._oneflow_internal.profiler.RangePop()
        flow._oneflow_internal.profiler.RangePush('.numpy')
        out = pred.numpy()
        flow._oneflow_internal.profiler.RangePop()
    flow._oneflow_internal.profiler.RangePop()
    t2 = time_sync()
print("time of inference: ", t2 - t1)