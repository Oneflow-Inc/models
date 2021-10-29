import numpy as np
import oneflow as flow


class Accuracy(flow.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, labels):
        top1_num = flow.zeros(1, dtype=flow.float32)
        num_samples = 0
        for pred, label in zip(preds, labels):
            clsidxs = pred.argmax(dim=-1)
            clsidxs = clsidxs.to(flow.int32)
            match = (clsidxs == label).sum()
            top1_num += match.to(device=top1_num.device, dtype=top1_num.dtype)
            num_samples += np.prod(label.shape).item()

        top1_acc = top1_num / num_samples
        return top1_acc
