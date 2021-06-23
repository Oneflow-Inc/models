import oneflow.experimental as flow
import oneflow.experimental.nn as nn
from oneflow.experimental import Tensor
from typing import Type, Any, Callable, Union, List, Optional


class BasicStem(nn.Module):
    def __init__(self, in_planes: int, out_planes: int,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv7x7 = self.conv7x7_fn(in_planes, out_planes)
        self.bn = norm_layer(out_planes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding = 1)

    def conv7x7_fn(in_planes: int, out_planes: int, stride: int = 2, groups: int = 1, dilation: int = 1,
                   padding: int = 3) -> nn.Conv2d:
        """stem 7x7 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                         padding=padding, groups=groups, bias=False, dilation=dilation)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv7x7(x)
        x = self.relu(self.bn(x))
        return self.maxpool(x)


# class fpn(nn.Module):


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = norm_layer(out_planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = norm_layer(out_planes)
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
            in_planes: int,
            out_planes: int,
            stride: int = 1,
            stride_in_1x1: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(out_planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_planes, width, stride_in_1x1)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, out_planes * self.expansion)
        self.bn3 = norm_layer(out_planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    model_settings = {
        18: {
            'block': BasicBlock,
            'num_blocks': (2, 2, 2, 2)
        },
        34: {
            'block': BasicBlock,
            'num_blocks': (3, 4, 6, 3)
        },
        50: {
            'block': Bottleneck,
            'num_blocks': (3, 4, 6, 3)
        },
        101: {
            'block': Bottleneck,
            'num_blocks': (3, 4, 23, 3)
        }
    }

    def __init__(
            self,
            # block: Type[Union[BasicBlock, Bottleneck]],
            # layers: List[int],
            # depth: int,
            # freeze_stage: int,
            # num_classes: int,
            # in_channels: int,
            # out_channels: int,
            # stride_in_1x1: int = None,
            # res5_dilation: int = None,
            cfg,
            zero_init_residual: bool = False,
            # groups: int = 1,
            # width_per_group: int = 64,
            # replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        depth = cfg.DEPTH
        freeze_stage = cfg.FREEZE_STAGE
        out_features = cfg.OUT_FEATURES
        num_groups = cfg.NUM_GROUPS
        width_per_group = cfg.WIDTH_PER_GROUP
        bottleneck_channels = num_groups * width_per_group
        in_channels = cfg.STEM_OUT_CHANNELS
        out_channels = cfg.RES2_OUT_CHANNELS
        stride_in_1x1 = cfg.STRIDE_IN_1X1
        res5_dilation = cfg.RES5_DILATION
        weight_decay = cfg.WEIGHT_DECAY

        # stem output channel and resnet input channel
        self.inplanes = 64
        self.dilation = 1
        if depth not in self.model_settings:
            raise ValueError("Depth is not in the model_settings of Resnet.")
        block = self.model_settings[depth]['block']
        layers = self.model_settings[depth]['num_blocks']

        # if replace_stride_with_dilation is None:
        #     # each element in the tuple indicates if we should replace
        #     # the 2x2 stride with a dilated convolution instead
        #     replace_stride_with_dilation = [False, False, False]
        # if len(replace_stride_with_dilation) != 3:
        #     raise ValueError("replace_stride_with_dilation should be None "
        #                      "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = num_groups
        self.base_width = width_per_group
        self.stem = BasicStem(3, self.inplanes)
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn1 = norm_layer(self.inplanes)
        # self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
        #                                dilate=replace_stride_with_dilation[0])
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
        #                                dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        #                                dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AvgPool2d((7, 7))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.stages = []
        self.stage_names = []

        out_stage_idx = [
            {"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features if f != "stem"
        ]
        max_stage_idx = max(out_stage_idx)
        for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
            dilation = res5_dilation if stage_idx == 5 else 1
            first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
            stage_kargs = {
                "num_blocks": layers[idx],
                "stride": [first_stride] + [1] * (layers[idx] - 1),
                "in_channels": in_channels,
                "out_channels": out_channels,
                "trainable": False if idx < freeze_stage else True,
                "weight_decay": weight_decay
            }
            if depth in [18, 34]:
                stage_kargs["block"] = BasicBlock
            else:
                stage_kargs["bottleneck_channels"] = bottleneck_channels
                stage_kargs["stride_in_1x1"] = stride_in_1x1
                stage_kargs["dilation"] = dilation
                stage_kargs["num_groups"] = num_groups
                stage_kargs["block"] = Bottleneck
            stage = self.make_stage(**stage_kargs)
            name = "res" + str(idx + 2)
            self.stage_names.append(name)
            in_channels = out_channels
            out_channels *= 2
            bottleneck_channels *= 2
            self.stages.append(stage)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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

    def make_stage(
            self, block: Type[Union[BasicBlock, Bottleneck]], stride, num_blocks, in_channels, out_channels, **kwargs
    ):
        blocks = []
        for i in range(num_blocks):
            stride_per_block = stride[i]
            blocks.append(
                block(
                    in_planes=in_channels,
                    out_planes=out_channels,
                    stride=stride_per_block,
                    **kwargs)
            )
            in_channels = out_channels
        return nn.Sequential(*blocks)


    # def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
    #                 stride: int = 1, dilate: bool = False) -> nn.Sequential:
    #     norm_layer = self._norm_layer
    #     downsample = None
    #     previous_dilation = self.dilation
    #     if dilate:
    #         self.dilation *= stride
    #         stride = 1
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             conv1x1(self.inplanes, planes * block.expansion, stride),
    #             norm_layer(planes * block.expansion),
    #         )
    #
    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
    #                         self.base_width, previous_dilation, norm_layer))
    #     self.inplanes = planes * block.expansion
    #     for _ in range(1, blocks):
    #         layers.append(block(self.inplanes, planes, groups=self.groups,
    #                             base_width=self.base_width, dilation=self.dilation,
    #                             norm_layer=norm_layer))
    #
    #     return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        outputs = {}
        x = self.stem(x)
        for stage_idx, (name, stage) in enumerate(zip(self.stage_names, self.stages)):
            stage_name = "res{}".format(stage_idx + 2)
            x = stage(x)
            if name == stage_name:
                outputs[name] = x
        return outputs




# def _resnet(
#         arch: str,
#         block: Type[Union[BasicBlock, Bottleneck]],
#         layers: List[int],
#         **kwargs: Any
# ) -> ResNet:
#     model = ResNet(block, layers, **kwargs)
#     return model


def resnet(**kwargs: Any) -> ResNet:
    r"""ResNet-5
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    return ResNet(**kwargs)
