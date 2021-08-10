import oneflow as flow
from oneflow import Tensor


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    scores_inds = flow.argsort(scores, dim=0, descending=True)
    boxes = flow.F.gather(boxes, scores_inds, axis=0)
    _nms_op = (
        flow.builtin_op("nms")
        .Input("in")
        .Output("out")
        .Attr("iou_threshold", iou_threshold)
        .Attr("keep_n", -1)
        .Build()
    )
    keep = _nms_op(boxes)[0]
    index = flow.squeeze(flow.argwhere(keep), dim=[1])
    return flow.F.gather(scores_inds, index, axis=0)
