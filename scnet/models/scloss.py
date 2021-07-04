import oneflow.experimental as flow
import oneflow.experimental.nn as nn
from oneflow.experimental import Tensor
from typing import Type, Any, Callable, Union, List, Optional


class ccmp(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super(ccmp, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.avgpool2d = nn.AvgPool2d(kernel_size, stride, padding, ceil_mode)
    def forward(self, x):
        x = x.transpose(3, 1)
        x = self.avgpool2d(x)
        x = x.transpose(3, 1)
        return x


class scloss(nn.Module):

    def __init__(self, label, cnum=3, num_classes=8):
        super(scloss, self).__init__()
        self.cnum = cnum
        self.label = label
        self.avgpool = nn.AvgPool2d((7, 7))
        self.fc = nn.Linear(2048, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def loss_div(self, x):
        branch = x
        branch = flow.reshape(branch, shape=[branch.size(0), branch.size(1), branch.size(2) * branch.size(3)])
        branch = flow.softmax(branch, 2)
        branch = flow.reshape(branch, shape=[branch.size(0), branch.size(1), x.size(2), x.size(2)])
        branch = ccmp(kernel_size=(1, self.cnum), stride=(1, self.cnum))(branch)
        branch = flow.reshape(branch, shape=[branch.size(0), branch.size(1), branch.size(2) * branch.size(3)])
        loss_dis = 1.0 - 1.0 * flow.mean(flow.sum(branch, 2)) / self.cnum  # set margin = 3.0
        return loss_dis

    def loss_con(self, feature):
        return self.criterion(feature,self.label)
    def loss_pre(self, fc):
        return self.criterion(fc,self.label)

    def _forward_impl(self, x: Tensor, y: Tensor) -> Tensor:
        loss_div = self.loss_div(x)
        x = self.avgpool(x)
        x = flow.flatten(x, 1)
        x = self.fc(x)
        loss_con = self.loss_con(x)
        loss_pre = self.loss_pre(y)
        return loss_con + loss_pre + loss_div

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self._forward_impl(x, y)
