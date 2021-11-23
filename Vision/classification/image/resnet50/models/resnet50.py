import oneflow as flow
import oneflow.nn as nn
from oneflow import Tensor
from typing import Type, Any, Callable, Union, List, Optional


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, data_format: str = "NCHW"
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        data_format=data_format,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, data_format: str = "NCHW") -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, data_format=data_format)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        fuse_bn_relu=False,
        fuse_bn_add_relu=False,
        data_format="NCHW",
    ) -> None:
        super(Bottleneck, self).__init__()
        self.fuse_bn_relu = fuse_bn_relu
        self.fuse_bn_add_relu = fuse_bn_add_relu
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, data_format=data_format)

        if self.fuse_bn_relu:
            self.bn1 = nn.FusedBatchNorm2d(width, data_format=data_format)
            self.bn2 = nn.FusedBatchNorm2d(width, data_format=data_format)
        else:
            self.bn1 = norm_layer(width, data_format=data_format)
            self.bn2 = norm_layer(width, data_format=data_format)
            self.relu = nn.ReLU()

        self.conv2 = conv3x3(width, width, stride, groups, dilation, data_format=data_format)
        self.conv3 = conv1x1(width, planes * self.expansion, data_format=data_format)

        if self.fuse_bn_add_relu:
            self.bn3 = nn.FusedBatchNorm2d(planes * self.expansion, data_format=data_format)
        else:
            self.bn3 = norm_layer(planes * self.expansion, data_format=data_format)
            self.relu = nn.ReLU()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        
        if self.fuse_bn_relu:
            out = self.bn1(out, None)
        else:
            out = self.bn1(out)
            out = self.relu(out)

        out = self.conv2(out)

        if self.fuse_bn_relu:
            out = self.bn2(out, None)
        else:
            out = self.bn2(out)
            out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.fuse_bn_add_relu:
            out = self.bn3(out, identity)
        else:
            out = self.bn3(out)
            out += identity
            out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        fuse_bn_relu=False,
        fuse_bn_add_relu=False,
        channel_last=False,
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.fuse_bn_relu = fuse_bn_relu
        self.fuse_bn_add_relu = fuse_bn_add_relu
        self.channel_last = channel_last
        if self.channel_last:
            self.pad_input = True
        else:
            self.pad_input = False

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        if self.pad_input:
            channel_size = 4
        else:
            channel_size = 3
        if self.channel_last:
            self.data_format = "NHWC"
        else:
            self.data_format = "NCHW"
        self.conv1 = nn.Conv2d(
            channel_size, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, data_format=self.data_format
        )
        print(">>>>>> ", self.conv1.__repr__())
        print(">>>>>> ", self.conv1.weight._meta_repr())

        if self.fuse_bn_relu:
            self.bn1 = nn.FusedBatchNorm2d(self.inplanes, data_format=self.data_format)
        else:
            self.bn1 = self._norm_layer(self.inplanes, data_format=self.data_format)
            self.relu = nn.ReLU()
        print(">>>>>> ", self.bn1.__repr__())
        print(">>>>>> ", self.bn1.running_mean._meta_repr())
        if self.channel_last:
            self.maxpool = nn.LegacyMaxPool2d(kernel_size=3, stride=2, padding=1, data_format=self.data_format)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        if self.channel_last:
           self.avgpool = nn.LegacyAvgPool2d((7, 7), stride=(1, 1), data_format=self.data_format)
        else:
           self.avgpool = nn.AvgPool2d((7, 7), stride=(1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu", data_format=self.data_format)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, data_format=self.data_format),
                norm_layer(planes * block.expansion, data_format=self.data_format),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                fuse_bn_relu=self.fuse_bn_relu,
                fuse_bn_add_relu=self.fuse_bn_add_relu,
                data_format=self.data_format,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    fuse_bn_relu=self.fuse_bn_relu,
                    fuse_bn_add_relu=self.fuse_bn_add_relu,
                    data_format=self.data_format,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        print("image shape ", x.shape)
        if self.pad_input:
            if self.channel_last:
                # NHWC
                paddings = (0, 1)
            else:
                # NCHW
                paddings = (0, 0, 0, 0, 0, 1)
            x = flow._C.pad(x, pad=paddings, mode="constant", value=0)
        print("x shape after padding ", x.shape)
        x = self.conv1(x)
        print("x shape after conv1 ", x.shape)
        if self.fuse_bn_relu:
            x = self.bn1(x, None)
        else:
            x = self.bn1(x)
            x = self.relu(x)
        print("x shape after bn1", x.shape)
        x = self.maxpool(x)
        print("x shape after maxpool ", x.shape)

        x = self.layer1(x)
        print("x shape after layer1 ", x.shape)
        x = self.layer2(x)
        print("x shape after layer2 ", x.shape)
        x = self.layer3(x)
        print("x shape after layer3 ", x.shape)
        x = self.layer4(x)
        print("x shape after layer4 ", x.shape)

        x = self.avgpool(x)
        print("x shape after avgpool ", x.shape)
        x = flow.flatten(x, 1)
        print("x shape after flatten ", x.shape)
        x = self.fc(x)
        print("x shape after fc ", x.shape)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


def resnet50(**kwargs: Any) -> ResNet:
    r"""ResNet-5
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], **kwargs)
