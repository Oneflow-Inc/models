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
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.flow_utils import select_device, time_sync


model = DetectMultiBackend(weights=ROOT / 'yolov5s.pt', device='cuda:0', dnn=False, data=ROOT / 'data/coco128.yaml')
model = model.eval()

# t1 = time_sync()
np.random.seed(0)
im = flow.tensor(np.random.rand(1, 3, 640, 640)).to('cuda:0')
im = im.float()
flow._oneflow_internal.profiler.RangePush('yolov5')
for i in range(20):
    pred = model(im, augment=False, visualize=False)
flow._oneflow_internal.profiler.RangePop()
# t2 = time_sync()
# flow.save(model.state_dict(), "./yolov5_model")
# np.save("./pred_oneflow.npy", pred.data.cpu().numpy())
# print("time of inference: ", t2 - t1)