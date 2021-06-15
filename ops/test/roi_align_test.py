import unittest
from collections import OrderedDict

import numpy as np

import oneflow.experimental as flow
from oneflow.python.test.modules.test_util import GenArgList
from ops import RoIAlign


def _test_roi_align(test_case, device):
    input = flow.Tensor(
        np.random.randn(2, 3, 64, 64), dtype=flow.float32, device=flow.device(device)
    )

    random_img_idx = np.random.randint(low=0, high=2, size=(200, 1))
    random_box_idx = np.random.uniform(low=0, high=64 * 64, size=(200, 2))

    def get_h_w(idx1, idx2):
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        h1 = idx1 // 64
        w1 = idx1 % 64
        h2 = idx2 // 64
        w2 = idx2 % 64
        return [x / 2 for x in [h1, w1, h2, w2]]

    zipped = zip(random_box_idx[:, 0], random_box_idx[:, 1])
    concated = [get_h_w(idx1, idx2) for (idx1, idx2) in zipped]
    concated = np.array(concated)
    rois = np.hstack((random_img_idx, concated))

    labels = np.random.randint(low=0, high=2, size=(200, 3, 14, 14)).astype(np.float32)

    roi_align_module = RoIAlign((14, 14), 2.0, 2, True)
    of_out = roi_align_module(input, rois)
    print(of_out)


class TestRoIAlign(flow.unittest.TestCase):
    def test_roi_align(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_roi_align
        ]
        arg_dict["device"] = ["cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
