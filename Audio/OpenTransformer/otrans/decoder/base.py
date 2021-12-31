import oneflow as flow
import oneflow.nn as nn


class BaseDecoder(nn.Module):
    def __init__(self):
        super(BaseDecoder, self).__init__()

    def forward(self, targets, **kwargs):
        raise NotImplementedError

    def inference(self, tokens, **kargs):
        raise NotImplementedError
