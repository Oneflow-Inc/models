import oneflow as flow
import oneflow.nn as nn
from typing import Union, List, Dict, Any, cast

__all__ = [
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
]


def get_placement(device_type="cuda"):
    return flow.placement(device_type, {0: range(flow.env.get_world_size())})


class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism.
    The linear layer is defined as Y = XA + b, where A is parallelized along 
    the second dimension as A = [A_1, ..., A_p].
    Arguments:
        layer_idx: the layer index, which determines the placement.
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        init_method: method to initialize weights.
        activation: name of the activation function
        bias_gelu_fusion: whether fuse add bias and gelu.
    """

    def __init__(
        self, input_size, output_size, init_method=nn.init.xavier_normal_,
    ):
        super().__init__()

        # column parallel linear weight sbp sign S(1)
        self.weight = flow.nn.Parameter(
            flow.empty(
                (input_size, output_size),
                dtype=flow.float32,
                placement=get_placement(),
                sbp=flow.sbp.split(1),
            ),
        )
        init_method(self.weight)

        # column parallel linear bias sbp: [B, S(0)]
        self.bias = flow.nn.Parameter(
            flow.empty(
                (output_size,),
                dtype=flow.float32,
                placement=get_placement(),
                sbp=flow.sbp.split(0),
            )
        )
        flow.nn.init.zeros_(self.bias)

    def forward(self, x):
        # x sbp: B
        # x.grad sbp: P -> B
        x = x.to_consistent(grad_sbp=x.sbp)
        # matmul sbp sign: [S(0), B] x [B, S(1)] -> [S(0), S(1)]
        # x.grad sbp sign: [S(0), S(1)] x [B, S(0)] (weight.T) -> [S(0), P]
        x = flow.matmul(x, self.weight)
        # broadcast_add shape sign:
        # (input_size, output_size) + (output_size, ) = (input_size, output_size)
        # bias_add sbp sign: [S(0), S(1)] + [B, S(0)] = [S(0), S(1)]
        x = x + self.bias

        return x


class RowParallelLinear(flow.nn.Module):
    """Linear layer with row parallelism.
    The linear layer is defined as Y = XA + b, where A is parallelized along 
    the first dimension and X along its second dimension as:
                | A_1 |
                |  .  |
            A = |  .  |         X = [X_1, ..., X_p]
                |  .  |
                | A_p |
    Arguments:
        layer_idx: the layer index, which determines the placement.
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        init_method: method to initialize weights.
        output_dropout_prob: dropout probability of output. (Supporting bias and dropout fusion)
        bias_dropout_fusion: whether fuse add bias and dropout.
    """

    def __init__(
        self, input_size, output_size, init_method=nn.init.xavier_normal_,
    ):
        super().__init__()

        # row parallel linear weight sbp: [B, S(0)]
        self.weight = flow.nn.Parameter(
            flow.empty(
                (input_size, output_size),
                dtype=flow.float32,
                placement=get_placement(),
                sbp=flow.sbp.split(0),
            )
        )
        init_method(self.weight)

        # row parallel linear bias sbp: [B, B]
        self.bias = flow.nn.Parameter(
            flow.empty(
                (output_size,),
                dtype=flow.float32,
                placement=get_placement(),
                sbp=flow.sbp.broadcast,
            ),
        )
        flow.nn.init.zeros_(self.bias)

    def forward(self, x):
        # x.sbp: [S(0), S(1)]
        # matmul sbp sign: [S(0), S(1)] x [B, S(0)] -> [S(0), P]
        # backward x.grad sbp sign: [S(0), B] x [B, S(1)] (weight.T) -> [S(0), S(1)]
        x = flow.matmul(x, self.weight)
        # x.sbp: [S(0), P] -> [S(0), B]
        x = x.to_consistent(sbp=flow.sbp.broadcast)
        x = x + self.bias

        return x


class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features.to_consistent(
            placement=get_placement(), sbp=flow.sbp.broadcast
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # column split A --> [A_1, A_2, ..., A_p]
        self.fc1 = ColumnParallelLinear(512 * 7 * 7, 4096)
        # row split
        #     | A_1 |
        #     |  .  |
        # A = |  .  |
        #     |  .  |
        #     | A_p |
        self.fc2 = RowParallelLinear(4096, 4096)
        # column split
        self.classifier = ColumnParallelLinear(4096, num_classes)

        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout()
        if init_weights:
            self._initialize_weights()

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = flow.flatten(x, 1)
        # NOTE(lxy): all_gather input from other device
        x = x.to_consistent(placement=x.placement, sbp=flow.sbp.broadcast)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # NOTE(lxy): because x's sbp sign is P, so all reduce for relu
        x = x.to_consistent(placement=x.placement, sbp=flow.sbp.broadcast)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def _vgg(arch: str, cfg: str, batch_norm: bool) -> VGG:
    return VGG(make_layers(cfgs[cfg], batch_norm=batch_norm))


def vgg16() -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg("vgg16", "D", False)


def vgg16_bn() -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg("vgg16_bn", "D", True)


def vgg19() -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg("vgg19", "E", False)


def vgg19_bn(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg("vgg19_bn", "E", True)
