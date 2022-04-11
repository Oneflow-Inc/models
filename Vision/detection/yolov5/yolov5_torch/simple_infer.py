import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.torch_utils import time_sync

model = DetectMultiBackend(weights=ROOT / 'yolov5s.pt', device='cuda:0', dnn=False, data=ROOT / 'data/coco128.yaml')
model = model.eval()

# t1 = time_sync()
np.random.seed(0)
im = torch.tensor(np.random.rand(1, 3, 640, 640)).to('cuda:0')
im = im.float()
with torch.no_grad():
    # 预热
    for i in range(5):
        pred = model(im, augment=False, visualize=False)
        out = pred.cpu().numpy()
    t1 = time_sync()
    # 埋点
    torch.cuda.nvtx.range_push("torch-yolov5-infer")
    for i in range(8000):
        torch.cuda.nvtx.range_push("forward")
        pred = model(im, augment=False, visualize=False)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push(".numpy")
        out = pred.cpu().numpy()
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()
    t2 = time_sync()
print("time of inference: ", t2 - t1)