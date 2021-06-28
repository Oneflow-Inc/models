import oneflow.experimental as flow
from oneflow.experimental import nn, Tensor

__all__ = [
    "create_RepVGG_A0",
    "create_RepVGG_A1",
    "create_RepVGG_A2",
    "create_RepVGG_B0",
    "create_RepVGG_B1",
    "create_RepVGG_B1g2",
    "create_RepVGG_B1g4",
    "create_RepVGG_B2",
    "create_RepVGG_B2g2",
    "create_RepVGG_B2g4",
    "create_RepVGG_B3",
    "create_RepVGG_B3g2",
    "create_RepVGG_B3g4",
    "create_RepVGG_D2se",
]


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module(
        "conv",
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
    )
    result.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
    return result


class SEBlock(nn.Module):
    def __init__(self, input_channels: int, internal_neurons: int):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(
            in_channels=input_channels,
            out_channels=internal_neurons,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.up = nn.Conv2d(
            in_channels=internal_neurons,
            out_channels=input_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.relu = nn.ReLU()
        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, inputs: Tensor):
        x = self.adaptive_avg_pool2d(inputs)
        x = self.down(x)
        x = self.relu(x)
        x = self.up(x)
        x = flow.sigmoid(x)
        return inputs * x


class RepVGGBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        padding_mode="zeros",
        deploy=False,
        use_se=False,
    ) -> None:
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode,
            )
        else:
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )
            self.rbr_dense = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
            )
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_11,
                groups=groups,
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.non_linearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(
            self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
        )


class RepVGG(nn.Module):
    def __init__(
        self,
        num_blocks,
        num_classes=1000,
        width_multiplier=None,
        override_groups_map=None,
        deploy=False,
        use_se=False,
    ):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.use_se = use_se

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            deploy=self.deploy,
            use_se=self.use_se,
        )
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(
            int(64 * width_multiplier[0]), num_blocks[0], stride=2
        )
        self.stage2 = self._make_stage(
            int(128 * width_multiplier[1]), num_blocks[1], stride=2
        )
        self.stage3 = self._make_stage(
            int(256 * width_multiplier[2]), num_blocks[2], stride=2
        )
        self.stage4 = self._make_stage(
            int(512 * width_multiplier[3]), num_blocks[3], stride=2
        )
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(
                RepVGGBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=cur_groups,
                    deploy=self.deploy,
                    use_se=self.use_se,
                )
            )
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = flow.flatten(out, 1)
        out = self.linear(out)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}


def create_RepVGG_A0(deploy=False):
    return RepVGG(
        num_blocks=[2, 4, 14, 1],
        num_classes=1000,
        width_multiplier=[0.75, 0.75, 0.75, 2.5],
        override_groups_map=None,
        deploy=deploy,
    )


def create_RepVGG_A1(deploy=False):
    return RepVGG(
        num_blocks=[2, 4, 14, 1],
        num_classes=1000,
        width_multiplier=[1, 1, 1, 2.5],
        override_groups_map=None,
        deploy=deploy,
    )


def create_RepVGG_A2(deploy=False):
    return RepVGG(
        num_blocks=[2, 4, 14, 1],
        num_classes=1000,
        width_multiplier=[1.5, 1.5, 1.5, 2.75],
        override_groups_map=None,
        deploy=deploy,
    )


def create_RepVGG_B0(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[1, 1, 1, 2.5],
        override_groups_map=None,
        deploy=deploy,
    )


def create_RepVGG_B1(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=None,
        deploy=deploy,
    )


def create_RepVGG_B1g2(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=g2_map,
        deploy=deploy,
    )


def create_RepVGG_B1g4(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=g4_map,
        deploy=deploy,
    )


def create_RepVGG_B2(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=None,
        deploy=deploy,
    )


def create_RepVGG_B2g2(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=g2_map,
        deploy=deploy,
    )


def create_RepVGG_B2g4(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=g4_map,
        deploy=deploy,
    )


def create_RepVGG_B3(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=None,
        deploy=deploy,
    )


def create_RepVGG_B3g2(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=g2_map,
        deploy=deploy,
    )


def create_RepVGG_B3g4(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=g4_map,
        deploy=deploy,
    )


def create_RepVGG_D2se(deploy=False):
    return RepVGG(
        num_blocks=[8, 14, 24, 1],
        num_classes=1000,
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=None,
        deploy=deploy,
        use_se=True,
    )
