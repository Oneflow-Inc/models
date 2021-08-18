import oneflow as flow
from oneflow.nn.module import Module
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class LabelSmooth(Module):
    def __init__(self, num_classes=-1, smooth_rate=0.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth_rate = smooth_rate
    
    def forward(self, input, label):
        _label = flow.Tensor(label, dtype=flow.int32)
        onehot_label = flow.nn.functional.one_hot(_label, num_classes= self.num_classes, 
                                                on_value=1-self.smooth_rate, 
                                                off_value=self.smooth_rate/(self.num_classes-1))
        log_prob = input.softmax(dim=-1).log()
        onehot_label = onehot_label.to(log_prob)
        loss = flow.mul(log_prob * -1, onehot_label).sum(dim=-1).mean()
        return loss

if __name__ == "__main__":
    for i in range(10):
        np_input = np.random.randn(3,3).astype(np.float32)
        np_label = np.array([0, 1, 2]).astype(np.int64)

        #oneflow
        flow_input = flow.Tensor(np_input.copy(), dtype=flow.float32)
        flow_label = flow.Tensor(np_label.copy(), dtype=flow.int64)
        loss = LabelSmooth(num_classes=3, smooth_rate=0.1)
        flow_loss = loss(flow_input, flow_label)

        #pytorch
        torch_input = torch.from_numpy(np_input.copy()).to(torch.float32)
        torch_label = torch.from_numpy(np_label.copy()).to(torch.int64)
        loss = LabelSmoothingLoss(classes=3, smoothing=0.1)
        torch_loss = loss(torch_input, torch_label)

        print(f"abs diff: {flow_loss.numpy() - torch_loss.numpy()}, flow labelsmooth loss: {flow_loss}, torch labelsmooth loss:{torch_loss}")