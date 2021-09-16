import oneflow as flow
import oneflow as flow_exp
from oneflow import Tensor


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    scores_inds = flow_exp.argsort(scores, dim=0, descending=True)
    boxes = flow._C.gather(boxes, scores_inds, axis=0)
    _nms_op = (
        flow_exp.builtin_op("nms")
        .Input("in")
        .Output("out")
        .Attr("iou_threshold", iou_threshold)
        .Attr("keep_n", -1)
        .Build()
    )
    keep = _nms_op(boxes)[0]
    index = flow_exp.squeeze(flow_exp.argwhere(keep), dim=[1])
    return flow._C.gather(scores_inds, index, axis=0)
