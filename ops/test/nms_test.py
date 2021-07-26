import unittest
from collections import OrderedDict

import numpy as np

import oneflow as flow
from ops.test.test_util import GenArgList
from ops import nms, lib_path

flow.config.load_library_now(lib_path())


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _box_inter_union_np(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = np.maximum(boxes1[:, np.newaxis, :2], boxes2[:, :2])
    rb = np.minimum(boxes1[:, np.newaxis, 2:], boxes2[:, 2:])

    wh = np.clip(rb - lt, a_min=0, a_max=np.inf)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, np.newaxis] + area2 - inter

    return inter, union


def box_iou_np(boxes1, boxes2):
    inter, union = _box_inter_union_np(boxes1, boxes2)
    iou = inter / union
    return iou


def nms_np(boxes, scores, iou_threshold):
    picked = []
    indexes = np.argsort(-scores)
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = np.squeeze(box_iou_np(rest_boxes, current_box[np.newaxis]), axis=1)
        indexes = indexes[iou <= iou_threshold]

    return np.asarray(picked)


def create_tensors_with_iou(N, iou_thresh):
    boxes = np.random.rand(N, 4) * 100
    boxes[:, 2:] += boxes[:, :2]
    boxes[-1, :] = boxes[0, :]
    x0, y0, x1, y1 = boxes[-1].tolist()
    iou_thresh += 1e-5
    boxes[-1, 2] += (x1 - x0) * (1 - iou_thresh) / iou_thresh
    scores = np.random.rand(N)
    return boxes, scores


def _test_nms(test_case, device):
    iou = 0.5
    boxes, scores = create_tensors_with_iou(1000, iou)
    boxes = flow.Tensor(boxes, dtype=flow.float32, device=flow.device(device))
    scores = flow.Tensor(scores, dtype=flow.float32, device=flow.device(device))
    keep_np = nms_np(boxes.numpy(), scores.numpy(), iou)
    keep = nms(boxes, scores, iou)
    test_case.assertTrue(np.allclose(keep.numpy(), keep_np))


class TestNMS(flow.unittest.TestCase):
    def test_nms(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_nms]
        arg_dict["device"] = ["cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
