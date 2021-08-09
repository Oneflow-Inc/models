from collections import OrderedDict
from time import sleep

from models.rpn import RPNHead, RegionProposalNetwork
from utils.anchor_utils import AnchorGenerator
import oneflow as flow
from models.faster_rcnn import TwoMLPHead
from oneflow import nn
# from ops import lib_path
# flow.config.load_library_now(lib_path())

import numpy as np
from utils.image_list import ImageList
flow.enable_eager_execution()
flow.InitEagerGlobalSession()


def _init_test_rpn():
    batch_size_per_image = 512
    bbox_reg_weights = None
    bg_iou_thresh = 0.5
    box_head = TwoMLPHead()
    box_predictor = {FastRCNNPredictor}
    FastRCNNPredictor(\n(cls_score): Linear(in_features=1024, out_features=5, bias=True)\n(bbox_pred): Linear(
        in_features=1024, out_features=20, bias=True)\n)
    box_roi_pool = {MultiScaleRoIAlign}
    MultiScaleRoIAlign()
    detections_per_img = {int}
    100
    fg_iou_thresh = {float}
    0.5
    keypoint_head = {NoneType}
    None
    keypoint_predictor = {NoneType}
    None
    keypoint_roi_pool = {NoneType}
    None
    mask_head = {NoneType}
    None
    mask_predictor = {NoneType}
    None
    mask_roi_pool = {NoneType}
    None
    nms_thresh = {float}
    0.5
    positive_fraction = {float}
    0.25
    score_thresh = {float}
    0.05

    rpn = RegionProposalNetwork(
        rpn_anchor_generator, rpn_head,
        rpn_fg_iou_thresh, rpn_bg_iou_thresh,
        rpn_batch_size_per_image, rpn_positive_fraction,
        rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
        score_thresh=rpn_score_thresh)
    return rpn

def get_features(images):
    s0, s1 = images.shape[-2:]
    features = [
        ('0', flow.Tensor(np.random.rand(2, 256, s0 // 4, s1 // 4))),
        ('1', flow.Tensor(np.random.rand(2, 256, s0 // 8, s1 // 8))),
        ('2', flow.Tensor(np.random.rand(2, 256, s0 // 16, s1 // 16))),
        ('3', flow.Tensor(np.random.rand(2, 256, s0 // 32, s1 // 32))),
        ('4', flow.Tensor(np.random.rand(2, 256, s0 // 64, s1 // 64))),
    ]

    features = OrderedDict(features)
    return features


def test_rpn():
    class RPNModule(nn.Module):
        def __init__(self):
            super(RPNModule, self).__init__()
            self.rpn = _init_test_rpn()

        def forward(self, images, features):
            # image_size = []
            # for i in images:
            #     image_size.append(i.shape[-2:])
            # images = ImageList(images, image_size)
            images = ImageList(images, [i.shape[-2:] for i in images])
            return self.rpn(images, features)

    images = flow.Tensor(np.random.rand(2, 3, 150, 150))
    features = get_features(images)
    images2 = flow.Tensor(np.random.rand(2, 3, 80, 80))
    test_features = get_features(images2)

    model = RPNModule()
    model.eval()
    res1 = model(images, features)
    res2 = model(images2, test_features)
    print(res1, res2)

    # self.run_model(model, [(images, features), (images2, test_features)], tolerate_small_mismatch=True,
    #                input_names=["input1", "input2", "input3", "input4", "input5", "input6"],
    #                dynamic_axes={"input1": [0, 1, 2, 3], "input2": [0, 1, 2, 3],
    #                              "input3": [0, 1, 2, 3], "input4": [0, 1, 2, 3],
    #                              "input5": [0, 1, 2, 3], "input6": [0, 1, 2, 3]})


if __name__ == '__main__':
    test_rpn()
