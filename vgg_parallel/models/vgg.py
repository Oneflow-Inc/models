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


class Linear1D(nn.Module):
    """Linear layer with 1D sbp"""

    def __init__(
        self,
        input_size,
        output_size,
        parallel="data",
        placement=get_placement(),
        init_method=nn.init.xavier_normal_,
    ):
        super().__init__()

        if parallel == "col":
            # column parallel linear weight sbp: S(1)
            w_sbp = flow.sbp.split(1)
            b_sbp = flow.sbp.split(0)
        elif parallel == "row":
            # row parallel linear weight sbp: S(0)
            w_sbp = flow.sbp.split(0)
            b_sbp = flow.sbp.broadcast
        elif parallel == "data":
            w_sbp = flow.sbp.broadcast
            b_sbp = flow.sbp.broadcast
        else:
            raise KeyError(
                f"{parallel} is not supported! Only support ('data', 'row' and 'col')"
            )

        self.weight = flow.nn.Parameter(
            flow.empty(
                (input_size, output_size),
                dtype=flow.float32,
                placement=placement,
                sbp=w_sbp,
            ),
        )
        self.bias = flow.nn.Parameter(
            flow.empty(
                (output_size,), dtype=flow.float32, placement=placement, sbp=b_sbp,
            )
        )

        init_method(self.weight)
        flow.nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.weight.sbp == flow.sbp.split(1):
            # 限定 x sbp: B，这样可以确保一定进行的是 tensor s(1) 切分
            x = x.to_consistent(sbp=flow.sbp.broadcast)
        elif self.weight.sbp == flow.sbp.split(0):
            #  限定 x.sbp: S(1), 确保一定进行的是 tensor s(0) 划分
            x = x.to_consistent(sbp=flow.sbp.split(1))
        elif self.weight.sbp == flow.sbp.broadcast:
            x = x.to_consistent(sbp=flow.sbp.split(0))
        # x.grad sbp: P -> B
        # x = x.to_consistent(grad_sbp=x.sbp)
        # matmul sbp sign: [S(0), B] x [B, S(1)] -> [S(0), S(1)]
        # x.grad sbp sign: [S(0), S(1)] x [B, S(0)] (weight.T) -> [S(0), P]
        x = flow.matmul(x, self.weight)
        # broadcast_add shape sign:
        # (input_size, output_size) + (output_size, ) = (input_size, output_size)
        # bias_add sbp sign: [S(0), S(1)] + [B, S(0)] = [S(0), S(1)]
        if self.weight.sbp == flow.sbp.split(1):
            x = x.to_consistent(sbp=flow.sbp.broadcast)
        x = x + self.bias

        return x


class VGG(nn.Module):
    def __init__(
        self,
        parallel_way,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True,
    ) -> None:
        super(VGG, self).__init__()
        self.features = features.to_consistent(
            placement=get_placement(), sbp=flow.sbp.broadcast
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            Linear1D(512 * 7 * 7, 4096, parallel_way[0]),
            # nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            Linear1D(4096, 4096, parallel_way[1]),
            # nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            Linear1D(4096, num_classes, parallel_way[2]),
            # nn.Linear(4096, num_classes)
        )
        # self.classifier = self.classifier.to_consistent(placement=get_placement(), sbp=flow.sbp.broadcast)
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = flow.flatten(x, 1)
        return self.classifier(x)

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


def _vgg(
    parallel_way: str, arch: str, cfg: str, batch_norm: bool, num_classes=1000
) -> VGG:
    return VGG(parallel_way, make_layers(cfgs[cfg], batch_norm=batch_norm), num_classes)


def vgg16() -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg("vgg16", "D", False)


def vgg16_bn(parallel_way, num_classes) -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg(parallel_way, "vgg16_bn", "D", True, num_classes=num_classes)


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
