import oneflow as flow
import oneflow.nn as nn


class BaseFrontEnd(nn.Module):
    def __init__(self):
        super(BaseFrontEnd, self).__init__()

    def forward(self, inputs, inputs_mask):
        raise NotImplementedError

    def inference(self, inputs, inputs_mask):
        return self.forward(inputs, inputs_mask)