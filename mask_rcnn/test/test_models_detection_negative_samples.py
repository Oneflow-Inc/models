# import torch
#
# import torchvision.models
# from torchvision.ops import MultiScaleRoIAlign
# from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
# from torchvision.models.detection.roi_heads import RoIHeads
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead

import oneflow.experimental as flow
from oneflow import nn, Tensor
import numpy as np
from models.faster_rcnn import FastRCNNPredictor, TwoMLPHead

from utils.anchor_utils import AnchorGenerator
from models.rpn import RPNHead, RegionProposalNetwork
from ops.ms_roi_align import MultiScaleRoIAlign
from models.roi_heads import RoIHeads
from models.faster_rcnn import fasterrcnn_resnet50_fpn

from models.mask_rcnn import maskrcnn_resnet50_fpn


def assertEqual(param, param1):
    assert param == param1


# def _make_empty_sample(add_masks=False, add_keypoints=False):
#     images = [flow.Tensor(np.random.rand(3, 100, 100), dtype=flow.float32)]
#     boxes = flow.zeros((0, 4), dtype=flow.float32)
#     negative_target = {"boxes": boxes,
#                        "labels": flow.zeros(0, dtype=flow.int64),
#                        "image_id": 4,
#                        "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
#                        "iscrowd": flow.zeros((0,), dtype=flow.int64)}
#
#     if add_masks:
#         negative_target["masks"] = flow.zeros(0, 100, 100, dtype=flow.uint8)
#
#     if add_keypoints:
#         negative_target["keypoints"] = flow.zeros(17, 0, 3, dtype=flow.float32)
#
#     targets = [negative_target]
#     return images, targets


def _make_empty_sample(add_masks=False, add_keypoints=False):
    images = [flow.Tensor(np.random.rand(3, 100, 100), dtype=flow.float32)]
    # boxes = flow.zeros((0, 4), dtype=flow.float32)
    boxes = flow.Tensor(np.sort(np.random.randint(0, 100, (2, 4))), dtype=flow.float32)
    negative_target = {"boxes": boxes,
                       "labels": flow.Tensor(np.random.rand(1), dtype=flow.int64),
                       "image_id": 4,
                       "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                       "iscrowd": flow.zeros(1, dtype=flow.int64)}

    if add_masks:
        negative_target["masks"] = flow.Tensor(np.random.rand(1, 100, 100), dtype=flow.uint8)

    if add_keypoints:
        negative_target["keypoints"] = flow.Tensor(np.random.rand(17, 0, 3), dtype=flow.float32)

    targets = [negative_target]
    return images, targets


def test_targets_to_anchors():
    _, targets = _make_empty_sample()
    anchors = [flow.Tensor(np.random.randint(-50, 50, (3, 4)), dtype=flow.float32)]

    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )
    rpn_head = RPNHead(4, rpn_anchor_generator.num_anchors_per_location()[0])

    head = RegionProposalNetwork(
        rpn_anchor_generator, rpn_head,
        0.5, 0.3,
        256, 0.5,
        2000, 2000, 0.7, 0.05)

    labels, matched_gt_boxes = head.assign_targets_to_anchors(anchors, targets)

    assertEqual(labels[0].sum(), 0)
    assertEqual(labels[0].shape, flow.Size([anchors[0].shape[0]]))
    assertEqual(labels[0].dtype, flow.float32)

    assertEqual(matched_gt_boxes[0].sum(), 0)
    assertEqual(matched_gt_boxes[0].shape, anchors[0].shape)
    assertEqual(matched_gt_boxes[0].dtype, flow.float32)


def test_assign_targets_to_proposals():
    proposals = [flow.Tensor(np.random.randint(-50, 50, (20, 4)), dtype=flow.float32)]
    gt_boxes = [flow.zeros((0, 4), dtype=flow.float32)]
    gt_labels = [flow.Tensor([[0]], dtype=flow.int64)]

    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2)

    resolution = box_roi_pool.output_size[0]
    representation_size = 1024
    box_head = TwoMLPHead(
        4 * resolution ** 2,
        representation_size)

    representation_size = 1024
    box_predictor = FastRCNNPredictor(
        representation_size,
        2)

    roi_heads = RoIHeads(
        # Box
        box_roi_pool, box_head, box_predictor,
        0.5, 0.5,
        512, 0.25,
        None,
        0.05, 0.5, 100)

    matched_idxs, labels = roi_heads.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)

    assertEqual(matched_idxs[0].sum(), 0)
    assertEqual(matched_idxs[0].shape, flow.Size([proposals[0].shape[0]]))
    assertEqual(matched_idxs[0].dtype, flow.int64)

    assertEqual(labels[0].sum(), 0)
    assertEqual(labels[0].shape, flow.Size([proposals[0].shape[0]]))
    assertEqual(labels[0].dtype, flow.int64)


def test_forward_negative_sample_frcnn():
    # for name in ["fasterrcnn_resnet50_fpn", "fasterrcnn_mobilenet_v3_large_fpn",
    #              "fasterrcnn_mobilenet_v3_large_320_fpn"]:
    #     model = torchvision.models.detection.__dict__[name](
    #         num_classes=2, min_size=100, max_size=100)
    model = fasterrcnn_resnet50_fpn(num_classes=2, min_size=100, max_size=100)
    images, targets = _make_empty_sample()
    loss_dict = model(images, targets)

    assertEqual(loss_dict["loss_box_reg"], flow.tensor(0.))
    assertEqual(loss_dict["loss_rpn_box_reg"], flow.tensor(0.))


def test_forward_negative_sample_mrcnn():
    model = maskrcnn_resnet50_fpn(
        num_classes=2, min_size=100, max_size=100)

    images, targets = _make_empty_sample(add_masks=True)
    loss_dict = model(images, targets)

    assertEqual(loss_dict["loss_box_reg"], flow.tensor(0.))
    assertEqual(loss_dict["loss_rpn_box_reg"], flow.tensor(0.))
    assertEqual(loss_dict["loss_mask"], flow.tensor(0.))

    # def test_forward_negative_sample_krcnn(self):
    #     model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
    #         num_classes=2, min_size=100, max_size=100)
    #
    #     images, targets = self._make_empty_sample(add_keypoints=True)
    #     loss_dict = model(images, targets)
    #
    #     self.assertEqual(loss_dict["loss_box_reg"], torch.tensor(0.))
    #     self.assertEqual(loss_dict["loss_rpn_box_reg"], torch.tensor(0.))
    #     self.assertEqual(loss_dict["loss_keypoint"], torch.tensor(0.))
    #
    # def test_forward_negative_sample_retinanet(self):
    #     model = torchvision.models.detection.retinanet_resnet50_fpn(
    #         num_classes=2, min_size=100, max_size=100, pretrained_backbone=False)
    #
    #     images, targets = self._make_empty_sample()
    #     loss_dict = model(images, targets)
    #
    #     self.assertEqual(loss_dict["bbox_regression"], torch.tensor(0.))


if __name__ == '__main__':
    # test_targets_to_anchors()
    # test_assign_targets_to_proposals()
    test_forward_negative_sample_mrcnn()
