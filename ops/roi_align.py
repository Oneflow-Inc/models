import oneflow as flow
import oneflow.nn as nn
from typing import List


class RoIAlign(nn.Module):
    def __init__(
        self,
        output_size: List[int],
        spatial_scale: float,
        sampling_ratio: int,
        aligned: bool = False,
    ):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned
        self._roi_align_op = (
            flow.builtin_op("roi_align")
            .Input("x")
            .Input("rois")
            .Output("y")
            .Attr("pooled_h", self.output_size[0])
            .Attr("pooled_w", self.output_size[1])
            .Attr("spatial_scale", spatial_scale)
            .Attr("sampling_ratio", sampling_ratio)
            .Attr("aligned", aligned)
            .Build()
        )

    def forward(self, input, rois):
        return self._roi_align_op(input, rois)[0]
