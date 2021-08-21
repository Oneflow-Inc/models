import numpy as np
import oneflow as flow
import oneflow.nn as nn
from oneflow import Tensor
from typing import Tuple
import  oneflow.F as F
import cv2
from Resnet50 import ResNet
from Resnet50 import Bottleneck


def weight_init(module):
    for n, m in module.named_children():
        print('initialize:' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity='relu')
            if m.bias is not None:
                flow.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):  # (nn.BatchNorm2d, flow.nn.InstanceNorm2d)
            nn.init.ones_(m.weight)
            if m.bias is not None:
                flow.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                flow.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.MaxPool2d)):
            pass
        else:
            m.initialize()


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv0 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu = flow.nn.ReLU(inplace=True)

    def forward(self, input1, input2=[0, 0, 0, 0]):
        out0 = F.relu(self.bn0(self.conv0(input1[0] + input2[0])), inplace=True)
        h = input1[1].size()[2:][0]
        w = input1[1].size()[2:][1]
        Size = (h, w)
        out0 = flow.nn.functional.interpolate(out0, size=Size, mode='bilinear')
        out1 = F.relu(self.bn1(self.conv1(input1[1] + input2[1] + out0)), inplace=True)
        h = input1[2].size()[2:][0]
        w = input1[2].size()[2:][1]
        Size = (h, w)
        out1 = flow.nn.functional.interpolate(out1, size=Size, mode='bilinear')
        out2 = F.relu(self.bn2(self.conv2(input1[2] + input2[2] + out1)), inplace=True)
        h = input1[3].size()[2:][0]
        w = input1[3].size()[2:][1]
        Size = (h, w)
        out2 = flow.nn.functional.interpolate(out2, size=Size, mode='bilinear')
        out3 = F.relu(self.bn3(self.conv3(input1[3] + input2[3] + out2)), inplace=True)
        return out3

    def initialize(self):
        weight_init(self)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.conv3b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3b = nn.BatchNorm2d(64)
        self.conv4b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4b = nn.BatchNorm2d(64)

        self.conv1d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1d = nn.BatchNorm2d(64)
        self.conv2d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2d = nn.BatchNorm2d(64)
        self.conv3d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3d = nn.BatchNorm2d(64)
        self.conv4d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4d = nn.BatchNorm2d(64)
        #self.relu = flow.nn.ReLU(inplace=True)
        self.maxpool = flow.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, out1):
        out1 = F.relu(self.bn1(self.conv1(out1)), inplace=True)
        out2 = self.maxpool(out1)
        out2 = F.relu(self.bn2(self.conv2(out2)), inplace=True)
        out3 = self.maxpool(out2)
        out3 = F.relu(self.bn3(self.conv3(out3)), inplace=True)
        out4 = self.maxpool(out3)
        out4 = F.relu(self.bn4(self.conv4(out4)), inplace=True)

        out1b = F.relu(self.bn1b(self.conv1b(out1)), inplace=True)
        out2b = F.relu(self.bn2b(self.conv2b(out2)), inplace=True)
        out3b = F.relu(self.bn3b(self.conv3b(out3)), inplace=True)
        out4b = F.relu(self.bn4b(self.conv4b(out4)), inplace=True)

        out1d = F.relu(self.bn1d(self.conv1d(out1)), inplace=True)
        out2d = F.relu(self.bn2d(self.conv2d(out2)), inplace=True)
        out3d = F.relu(self.bn3d(self.conv3d(out3)), inplace=True)
        out4d = F.relu(self.bn4d(self.conv4d(out4)), inplace=True)
        return (out4b, out3b, out2b, out1b), (out4d, out3d, out2d, out1d)

    def initialize(self):
        weight_init(self)


class LDF(nn.Module):
    def __init__(self, cfg):
        super(LDF, self).__init__()
        self.cfg = cfg
        self.model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], last_stride=1)
        self.conv5b = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4b = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3b = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2b = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv5d = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4d = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3d = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2d = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.encoder = Encoder()
        self.deconderb = Decoder()
        self.deconderd = Decoder()
        self.linearb = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.lineard = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linear = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=3))
        self.initialize()

    def forward(self, x, shape=None):
        out2, out3, out4, out5 = self.model(x)
        out2b, out3b, out4b, out5b = self.conv2b(out2), self.conv3b(out3), self.conv4b(out4), self.conv5b(out5)
        out2d, out3d, out4d, out5d = self.conv2d(out2), self.conv3d(out3), self.conv4d(out4), self.conv5d(out5)
        outb1 = self.deconderb([out5b, out4b, out3b, out2b])
        outd1 = self.deconderb([out5d, out4d, out3d, out2d])
        out1 = flow.cat([outb1, outd1], dim=1)

        outb2, outd2 = self.encoder(out1)

        outb2 = self.deconderb([out5b, out4b, out3b, out2b], outb2)

        outd2 = self.deconderd([out5d, out4d, out3d, out2d], outd2)
        out2 = flow.cat([outb2, outd2], dim=1)

        if shape is None:
            shape = x.size()[2:]
        Size = (shape[0], shape[1])
        out1 = flow.nn.functional.interpolate(self.linear(out1), size=Size, mode="bilinear")
        outb1 = flow.nn.functional.interpolate(self.linearb(outb1), size=Size, mode="bilinear")
        outd1 = flow.nn.functional.interpolate(self.lineard(outd1), size=Size, mode="bilinear")

        out2 = flow.nn.functional.interpolate(self.linear(out2), size=Size, mode="bilinear")
        outb2 = flow.nn.functional.interpolate(self.linearb(outb2), size=Size, mode="bilinear")
        outd2 = flow.nn.functional.interpolate(self.lineard(outd2), size=Size, mode="bilinear")

        return outb1, outd1, out1, outb2, outd2, out2

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(flow.load(self.cfg.snapshot))
        else:
            weight_init(self)

