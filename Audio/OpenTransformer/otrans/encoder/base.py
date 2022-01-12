import oneflow as flow
import oneflow.nn as nn


class BaseEncoder(nn.Module):
    def __init__(self):
        super(BaseEncoder, self).__init__()

    def forward(self, inputs, inputs_mask, **kargs):
        raise NotImplementedError

    def inference(self, inputs, inputs_mask, cache=None, **kargs):
        raise NotImplementedError
