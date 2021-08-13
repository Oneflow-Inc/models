import oneflow as flow
from oneflow.nn.module import Module
import numpy as np

# class LabelSmoothLoss(Module):
#     def __init__(self, smooth_rate: float = 0.0,) -> None:
#         super().__init__()
#         assert smooth_rate >= 0 and smooth_rate <= 1
#         self.smooth_rate = smooth_rate

#     def forward(self, input, target):
#         assert len(input.shape) == 2
#         assert len(target.shape) == len(input.shape) - 1
#         log_prob = input.softmax(dim=-1).log()
#         label = flow.zeros(input.size(), dtype=input.dtype) + self.smooth_rate / (input.size()[-1] - 1.)
#         #label = np.zeros(input.size()).astype("float32") + self.smooth_rate / (input.size()[-1] - 1.)
#         for k,class_idx in enumerate(target):
#             label[k, class_idx] = 1 - self.smooth_rate
#         return (-label * log_prob).sum(dim=-1).mean()

def one_hot(target, classes = 1000, smooth_rate=0.1):
    label = np.zeros((target.numpy().shape[0], classes)).astype("float32") + smooth_rate / (classes - 1.)
    for k,class_idx in enumerate(target):
        label[k, class_idx.numpy()] = 1 - smooth_rate
    return flow.tensor(label)

class LabelSmoothLoss(Module):
    def __init__(self,) -> None:
        super().__init__()
    
    def forward(self, input, label):
        assert label.size() == input.size()
        log_prob = input.softmax(dim=-1).log()
        loss = flow.mul(log_prob * -1, label).sum(dim=-1).mean()
        return loss


if __name__ == "__main__":
    input = flow.Tensor(
           [[-0.1664078, -1.7256707, -0.14690138],
            [-0.21474946, 0.53737473, 0.99684894],
            [-1.135804, -0.50371903, 0.7645404]], dtype=flow.float32)
    target = flow.Tensor([0, 1, 2], dtype=flow.int32)
    out = flow.nn.CrossEntropyLoss(reduction="none")(input, target)
    out_mean = flow.nn.CrossEntropyLoss(reduction="mean")(input, target)
    print(out_mean)
    label = one_hot(target, classes=3, smooth_rate=0.0)
    out_ls = LabelSmoothLoss()(input, label)
    print(out_ls)