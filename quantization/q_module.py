import math

import numpy as np
import oneflow as flow
import oneflow.nn as nn


class QParam:

    def __init__(self, quantization_bit=8, quantization_scheme='symmetric', quantization_formula='google', per_layer_quantization=True):
        self.quantization_bit = quantization_bit
        self.quantization_scheme = quantization_scheme
        self.quantization_formula = quantization_formula
        self.per_layer_quantization = per_layer_quantization
        self.scale = None
        self.zero_point = None

    def update(self, tensor):
        self.scale, self.zero_point = flow.quantization.min_max_observer(tensor, quantization_bit=self.quantization_bit, 
                                                                        quantization_scheme=self.quantization_scheme, per_layer_quantization=self.per_layer_quantization)

    def quantize_tensor(self, tensor):
        return flow.quantization.quantization(tensor, self.scale, self.zero_point, self.quantization_formula, self.quantization_bit, self.quantization_scheme)
    
    def fake_quantize_tensor(self, tensor):
        return flow.quantization.fake_quantization(tensor, self.scale, self.zero_point, self.quantization_formula, self.quantization_bit, self.quantization_scheme)

    def __str__(self):
        info = 'scale: %.10f ' % self.scale
        info += 'zp: %d ' % self.zero_point
        return info

class QModule(nn.Module):

    def __init__(self, qi=True, qo=True, quantization_bit=8, quantization_scheme='symmetric', quantization_formula='google', per_layer_quantization=True):
        super(QModule, self).__init__()
        if qi:
            self.qi = QParam(quantization_bit=quantization_bit, quantization_scheme=quantization_scheme, quantization_formula=quantization_formula, per_layer_quantization=per_layer_quantization)
        if qo:
            self.qo = QParam(quantization_bit=quantization_bit, quantization_scheme=quantization_scheme, quantization_formula=quantization_formula, per_layer_quantization=per_layer_quantization)

    def freeze(self):
        pass

    def quantize_inference(self, x):
        raise NotImplementedError('quantize_inference should be implemented.')

class QConv2d(QModule):

    def __init__(self, conv_module, qi=True, qo=True, quantization_bit=8, quantization_scheme='symmetric', quantization_formula='google', per_layer_quantization=True):
        super(QConv2d, self).__init__(qi=qi, qo=qo, quantization_bit=quantization_bit, quantization_scheme=quantization_scheme, quantization_formula=quantization_formula, per_layer_quantization=per_layer_quantization)
        self.quantization_bit = quantization_bit
        self.quantization_scheme = quantization_scheme
        self.quantization_formula = quantization_formula
        self.per_layer_quantization = per_layer_quantization
        self.conv_module = conv_module
        self.qw = QParam(quantization_bit=quantization_bit, quantization_scheme=quantization_scheme, quantization_formula=quantization_formula, per_layer_quantization=per_layer_quantization)

    def freeze(self, qi=None, qo=None):
        
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        self.M = self.qw.scale * self.qi.scale / self.qo.scale

        self.conv_module.weight.data = self.qw.quantize_tensor(self.conv_module.weight.data)
        self.conv_module.weight.data = self.conv_module.weight.data - self.qw.zero_point

        self.conv_module.bias.data = flow.quantization.fake_quantization(self.conv_module.bias.data, scale=self.qi.scale * self.qw.scale,
                                                     zero_point=0, quantization_formula="google", quantization_bit=32, quantization_scheme="affine")

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = self.qi.fake_quantize_tensor(x)

        self.qw.update(self.conv_module.weight.data)

        x = flow.F.conv2d(x, self.qw.fake_quantize_tensor(self.conv_module.weight), self.conv_module.bias, 
                     stride=self.conv_module.stride,
                     padding=self.conv_module.padding, dilation=self.conv_module.dilation, 
                     groups=self.conv_module.groups)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = self.qo.fake_quantize_tensor(x)

        return x

    def quantize_inference(self, x):
        x = x - self.qi.zero_point
        x = self.conv_module(x)
        x = self.M * x
        x = x.round() 
        x = x + self.qo.zero_point        
        x = x.clamp(0., 2.**self.quantization_bit-1.).round()
        return x


class QReLU(QModule):

    def __init__(self, qi=False):
        super(QReLU, self).__init__(qi=qi)

    def freeze(self, qi=None):
        
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if qi is not None:
            self.qi = qi

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = self.qi.fake_quantize_tensor(x)

        x = flow.F.relu(x)

        return x
    
    def quantize_inference(self, x):
        x = x.clone()
        x[x < self.qi.zero_point] = self.qi.zero_point
        return x

class QMaxPooling2d(QModule):

    def __init__(self, kernel_size=3, stride=1, padding=0, qi=False, quantization_bit=None):
        super(QMaxPooling2d, self).__init__(qi=qi, quantization_bit=quantization_bit)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def freeze(self, qi=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')
        if qi is not None:
            self.qi = qi

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = self.qi.fake_quantize_tensor(x)

        x = flow.F.max_pool2d(x, self.kernel_size, self.stride, self.padding)

        return x

    def quantize_inference(self, x):
        return flow.F.max_pool2d(x, self.kernel_size, self.stride, self.padding)

class QLinear(QModule):

    def __init__(self, fc_module, qi=True, qo=True, quantization_bit=8):
        super(QLinear, self).__init__(qi=qi, qo=qo, quantization_bit=quantization_bit)
        self.quantization_bit = quantization_bit
        self.fc_module = fc_module
        self.qw = QParam(quantization_bit=quantization_bit)

    def freeze(self, qi=None, qo=None):

        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        self.M = self.qw.scale * self.qi.scale / self.qo.scale

        self.fc_module.weight.data = self.qw.quantize_tensor(self.fc_module.weight.data)
        self.fc_module.weight.data = self.fc_module.weight.data - self.qw.zero_point
        self.fc_module.bias.data = flow.quantization.fake_quantization(self.fc_module.bias.data, scale=self.qi.scale * self.qw.scale,
                                                     zero_point=0, quantization_formula="google", quantization_bit=32, quantization_scheme="affine")


    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = self.qi.fake_quantize_tensor(x)

        self.qw.update(self.fc_module.weight.data)

        x = flow.F.linear(x, self.qw.fake_quantize_tensor(self.fc_module.weight), self.fc_module.bias)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = self.qo.fake_quantize_tensor(x)

        return x

    def quantize_inference(self, x):
        x = x - self.qi.zero_point
        x = self.fc_module(x)
        x = self.M * x
        x = x.round() 
        x = x + self.qo.zero_point
        x = x.clamp(0., 2.**self.quantization_bit-1.).round()
        return x

