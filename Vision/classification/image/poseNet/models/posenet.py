import oneflow as flow
from oneflow import nn

from typing import Any

__all__ = ["PoseNet", "posenet"]


class BasicConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Stem(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            BasicConv2d(input_channels, 32, kernel_size=3),
            BasicConv2d(32, 32, kernel_size=3, padding=1),
            BasicConv2d(32, 64, kernel_size=3, padding=1),
        )

        self.branch3x3_conv = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3_pool = nn.MaxPool2d(3, stride=1, padding=1)

        self.branch7x7a = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1),
            BasicConv2d(64, 64, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(64, 64, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
        )

        self.branch7x7b = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
        )

        self.branchpoola = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branchpoolb = BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1)

    def forward(self, x: flow.Tensor) -> flow.Tensor:

        x = self.conv1(x)

        x = [self.branch3x3_conv(x), self.branch3x3_pool(x)]
        x = flow.cat(x, 1)

        x = [self.branch7x7a(x), self.branch7x7b(x)]
        x = flow.cat(x, 1)

        x = [self.branchpoola(x), self.branchpoolb(x)]

        x = flow.cat(x, 1)

        return x


class Mixed_5b(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.Branch_2 = nn.Sequential(
            BasicConv2d(input_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1),
        )

        self.Branch_1 = nn.Sequential(
            BasicConv2d(input_channels, 48, kernel_size=1, padding=1),
            BasicConv2d(48, 64, kernel_size=5, padding=1),
        )

        self.Branch_0 = BasicConv2d(input_channels, 96, kernel_size=1)

        self.Branch_3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, 64, kernel_size=1),
        )

    def forward(self, input: flow.Tensor) -> flow.Tensor:
        x = input
        x = [self.Branch_0(x), self.Branch_1(x), self.Branch_2(x), self.Branch_3(x)]

        output = flow.cat(x, 1)

        return output


class block35(nn.Module):
    def __init__(self, input_channels):

        super().__init__()
        self.Branch_2 = nn.Sequential(
            BasicConv2d(input_channels, 32, kernel_size=1),
            BasicConv2d(32, 48, kernel_size=3, padding=1),
            BasicConv2d(48, 64, kernel_size=3, padding=1),
        )

        self.Branch_1 = nn.Sequential(
            BasicConv2d(input_channels, 32, kernel_size=1),
            BasicConv2d(32, 32, kernel_size=3, padding=1),
        )

        self.Branch_0 = BasicConv2d(input_channels, 32, kernel_size=1)

        self.Conv2d_1x1 = nn.Conv2d(128, 320, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, in_blob: flow.Tensor) -> flow.Tensor:
        residual = [
            self.Branch_0(in_blob),
            self.Branch_1(in_blob),
            self.Branch_2(in_blob),
        ]
        residual = flow.cat(residual, 1)

        up = self.Conv2d_1x1(residual)
        scaled_up = up * 1.0

        in_blob += scaled_up
        in_blob = self.relu(in_blob)

        return in_blob


class block17(nn.Module):
    def __init__(self, input_channels):

        super().__init__()
        self.Branch_1 = nn.Sequential(
            BasicConv2d(input_channels, 128, kernel_size=1, padding=1),
            BasicConv2d(128, 160, kernel_size=[1, 7], padding=1),
            BasicConv2d(160, 192, kernel_size=[7, 1], padding=1),
        )

        self.Branch_0 = BasicConv2d(input_channels, 192, kernel_size=1)
        self.Conv2d_1x1 = nn.Conv2d(384, input_channels, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, in_blob: flow.Tensor) -> flow.Tensor:

        residual = [self.Branch_0(in_blob), self.Branch_1(in_blob)]
        mixed = flow.cat(residual, 1)
        up = self.Conv2d_1x1(mixed)

        scaled_up = up * 1.0

        in_blob += scaled_up
        in_blob = self.relu(in_blob)

        return in_blob


class block8(nn.Module):
    def __init__(self, input_channels):

        super().__init__()
        self.Branch_1 = nn.Sequential(
            BasicConv2d(input_channels, 192, kernel_size=1),
            BasicConv2d(192, 224, kernel_size=[1, 3], padding=[0, 1]),
            BasicConv2d(224, 256, kernel_size=[3, 1], padding=[1, 0]),
        )

        self.Branch_0 = BasicConv2d(input_channels, 192, kernel_size=1)
        self.Conv2d_1x1 = BasicConv2d(448, 2080, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, in_blob: flow.Tensor) -> flow.Tensor:
        residual = [self.Branch_0(in_blob), self.Branch_1(in_blob)]
        mixed = flow.cat(residual, 1)
        up = self.Conv2d_1x1(mixed)

        scaled_up = up * 1.0

        in_blob += scaled_up
        in_blob = self.relu(in_blob)

        return in_blob


class Mixed_6a(nn.Module):
    def __init__(self, input_channels):

        super().__init__()
        self.Branch_2 = nn.MaxPool2d(3, stride=2)

        self.Branch_0 = nn.Sequential(
            BasicConv2d(input_channels, 384, kernel_size=3, stride=2)
        )

        self.Branch_1 = nn.Sequential(
            BasicConv2d(input_channels, 256, kernel_size=1),
            BasicConv2d(256, 256, kernel_size=3, padding=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2),
        )

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        x = [
            self.Branch_0(x),
            self.Branch_1(x),
            self.Branch_2(x),
        ]

        x = flow.cat(x, 1)
        return x


class Mixed_7a(nn.Module):
    def __init__(self, input_channels):

        super().__init__()
        self.Branch_0 = nn.Sequential(
            BasicConv2d(input_channels, 256, kernel_size=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2),
        )

        self.Branch_1 = nn.Sequential(
            BasicConv2d(input_channels, 256, kernel_size=1),
            BasicConv2d(256, 288, kernel_size=3, stride=2),
        )

        self.Branch_2 = nn.Sequential(
            BasicConv2d(input_channels, 256, kernel_size=1),
            BasicConv2d(256, 288, kernel_size=3, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2),
        )

        self.Branch_3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        x = [self.Branch_0(x), self.Branch_1(x), self.Branch_2(x), self.Branch_3(x)]

        x = flow.cat(x, 1)
        return x


class PoseNet(nn.Module):
    def __init__(self, num_classes: int = 5) -> None:
        super(PoseNet, self).__init__()

        self.conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.MaxPool_3a_3x3 = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.MaxPool_5a_3x3 = nn.MaxPool2d(kernel_size=3, stride=2)  # stem

        self.Mixed_5b = self._generate_inception_module(192, 320, 1, Mixed_5b)
        self.block35 = self._generate_inception_module(320, 320, 1, block35)

        self.conv_ls1 = BasicConv2d(320, 320, kernel_size=3, stride=2, padding=1)
        self.MaxPool_3x3_ls1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.Mixed_6a = self._generate_inception_module(320, 1088, 1, Mixed_6a)
        self.block17 = self._generate_inception_module(1088, 1088, 1, block17)

        self.conv_ls2 = BasicConv2d(1088, 1088, kernel_size=3, stride=2)

        self.Mixed_7a = self._generate_inception_module(1088, 2080, 1, Mixed_7a)
        self.block8 = self._generate_inception_module(2080, 2080, 1, block8)

        self.conv_ls3 = BasicConv2d(3488, 2080, kernel_size=1)
        self.Conv2d_7b_1x1 = BasicConv2d(2080, 1536, kernel_size=1)
        self.AvgPool_1a_8x8 = nn.AvgPool2d(kernel_size=[8, 8])

        self.dense = nn.Linear(1536, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: flow.Tensor) -> flow.Tensor:
        net = self.conv2d_1a_3x3(inputs)
        net = self.conv2d_2a_3x3(net)
        net = self.conv2d_2b_3x3(net)
        net = self.MaxPool_3a_3x3(net)
        net = self.conv2d_3b_1x1(net)
        net = self.conv2d_4a_3x3(net)
        net = self.MaxPool_5a_3x3(net)  # stem

        net = self.Mixed_5b(net)
        net = self.block35(net)

        netB1 = self.conv_ls1(net)
        netB1 = self.MaxPool_3x3_ls1(netB1)

        net = self.Mixed_6a(net)
        net = self.block17(net)

        netB2 = self.conv_ls2(net)
        net = self.Mixed_7a(net)
        net = self.block8(net)

        netB3 = [netB1, netB2, net]

        netAll = flow.cat(netB3, 1)
        netAll = self.conv_ls3(netAll)

        net = self.Conv2d_7b_1x1(netAll)
        net = self.AvgPool_1a_8x8(net)
        net = flow.reshape(net, [net.shape[0], -1])

        hidden = self.dense(net)
        hidden = self.relu(hidden)

        return hidden

    @staticmethod
    def _generate_inception_module(input_channels, output_channels, block_num, block):
        layers = nn.Sequential()
        for l in range(block_num):
            layers.add_module("{}_{}".format(block.__name__, l), block(input_channels))
            input_channels = output_channels

        return layers


def _posenet(arch: str, **kwargs: Any) -> PoseNet:
    model = PoseNet(**kwargs)
    return model


def posenet(**kwargs: Any) -> PoseNet:
    return _posenet("posenet", **kwargs)
