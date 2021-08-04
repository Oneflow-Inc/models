import oneflow as flow
import oneflow.nn as nn
from quantization_ops import *

__all__ = ["AlexNet"]

class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = flow.flatten(x, 1)
        x = self.classifier(x)
        return x

    def quantize(self, quantization_bit=8, quantization_scheme='symmetric', quantization_formula='google', per_layer_quantization=True):
        self.q_features = nn.Sequential(
            q_conv(self.features[0], qi=True, qo=False, quantization_bit=quantization_bit, quantization_scheme=quantization_scheme, quantization_formula=quantization_formula, per_layer_quantization=per_layer_quantization),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            q_conv(self.features[3], qi=True, qo=False, quantization_bit=quantization_bit, quantization_scheme=quantization_scheme, quantization_formula=quantization_formula, per_layer_quantization=per_layer_quantization),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            q_conv(self.features[6], qi=True, qo=False, quantization_bit=quantization_bit, quantization_scheme=quantization_scheme, quantization_formula=quantization_formula, per_layer_quantization=per_layer_quantization),
            nn.ReLU(inplace=True),
            q_conv(self.features[8], qi=True, qo=False, quantization_bit=quantization_bit, quantization_scheme=quantization_scheme, quantization_formula=quantization_formula, per_layer_quantization=per_layer_quantization),
            nn.ReLU(inplace=True),
            q_conv(self.features[10], qi=True, qo=False, quantization_bit=quantization_bit, quantization_scheme=quantization_scheme, quantization_formula=quantization_formula, per_layer_quantization=per_layer_quantization),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.q_avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.q_classifier = nn.Sequential(
            nn.Dropout(),
            q_linear(self.classifier[1], qi=True, qo=False, quantization_bit=quantization_bit, quantization_scheme=quantization_scheme, quantization_formula=quantization_formula, per_layer_quantization=per_layer_quantization),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            q_linear(self.classifier[4], qi=True, qo=False, quantization_bit=quantization_bit, quantization_scheme=quantization_scheme, quantization_formula=quantization_formula, per_layer_quantization=per_layer_quantization),
            nn.ReLU(inplace=True),
            q_linear(self.classifier[6], qi=True, qo=False, quantization_bit=quantization_bit, quantization_scheme=quantization_scheme, quantization_formula=quantization_formula, per_layer_quantization=per_layer_quantization),
        )

    def quantize_forward(self, x):
        x = self.q_features(x)
        x = self.q_avgpool(x)
        x = flow.flatten(x, 1)
        x = self.q_classifier(x)
        return x

    def freeze(self):
        self.q_features[0].freeze()
        self.q_features[3].freeze()
        self.q_features[6].freeze()
        self.q_features[8].freeze()
        self.q_features[10].freeze()
        self.q_classifier[1].freeze()
        self.q_classifier[4].freeze()
        self.q_classifier[6].freeze()

    def quantize_inference(self, x):
        pass
