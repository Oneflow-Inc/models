import oneflow as flow
import oneflow.nn as nn
from ..ops import *

class SimpleNet(nn.Module):

    def __init__(self, num_channels=3):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 6, 3, 1, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 120, 3, 1, 1)
        self.fc = nn.Linear(4*4*120, 10)

    def forward(self, x):
        x = flow.F.relu(self.conv1(x))
        x = flow.F.max_pool_2d(x, 2, 2)
        x = flow.F.relu(self.conv2(x))
        x = flow.F.max_pool_2d(x, 2, 2)
        x = flow.F.relu(self.conv3(x))
        x = flow.F.max_pool_2d(x, 2, 2)
        x = x.view(-1, 4*4*120)
        x = self.fc(x)
        return x

    def quantize(self, quantization_bit=8, quantization_scheme='symmetric', quantization_formula='google', per_layer_quantization=True):
        self.qconv1 = q_conv(self.conv1, qi=True, qo=True, quantization_bit=quantization_bit, quantization_scheme=quantization_scheme, quantization_formula=quantization_formula, per_layer_quantization=per_layer_quantization)
        self.qrelu1 = q_relu()
        self.qmaxpool2d_1 = q_max_pool_2d(kernel_size=2, stride=2, padding=0)
        self.qconv2 = q_conv(self.conv2, qi=False, qo=True, quantization_bit=quantization_bit, quantization_scheme=quantization_scheme, quantization_formula=quantization_formula, per_layer_quantization=per_layer_quantization)
        self.qrelu2 = q_relu()
        self.qmaxpool2d_2 = q_max_pool_2d(kernel_size=2, stride=2, padding=0)
        self.qfc = q_linear(self.fc, qi=False, qo=True, quantization_bit=quantization_bit, quantization_scheme=quantization_scheme, quantization_formula=quantization_formula, per_layer_quantization=per_layer_quantization)

    def quantize_forward(self, x):
        x = self.qconv1(x)
        x = self.qrelu1(x)
        x = self.qmaxpool2d_1(x)
        x = self.qconv2(x)
        x = self.qrelu2(x)
        x = self.qmaxpool2d_2(x)
        x = x.view(-1, 4*4*120)
        x = self.qfc(x)
        return x

    def freeze(self):
        self.qconv1.freeze()
        self.qrelu1.freeze(self.qconv1.qo)
        self.qmaxpool2d_1.freeze(self.qconv1.qo)
        self.qconv2.freeze(qi=self.qconv1.qo)
        self.qrelu2.freeze(self.qconv2.qo)
        self.qmaxpool2d_2.freeze(self.qconv2.qo)
        self.qfc.freeze(qi=self.qconv2.qo)

    def quantize_inference(self, x):
        qx = self.qconv1.qi.quantize_tensor(x)
        qx = self.qconv1.quantize_inference(qx)
        qx = self.qrelu1.quantize_inference(qx)
        qx = self.qmaxpool2d_1.quantize_inference(qx)
        qx = self.qconv2.quantize_inference(qx)
        qx = self.qrelu2.quantize_inference(qx)
        qx = self.qmaxpool2d_2.quantize_inference(qx)
        qx = qx.view(-1, 4*4*120)
        qx = self.qfc.quantize_inference(qx)
        out = self.qfc.qo.dequantize_tensor(qx)
        return out
