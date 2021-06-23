import oneflow.experimental as flow
import oneflow.experimental.nn as nn
from oneflow.experimental import Tensor
from typing import Type, Any, Callable, Union, List, Optional

class FPN(nn.module):
    def __init__(self, in_planes,
        out_planes,
        in_features,
        out_features,
        # weight_decay,
        top_block=None):
        super(FPN, self).__init__()
        self.lateral = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=1, bias=True, dilation=1)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)



