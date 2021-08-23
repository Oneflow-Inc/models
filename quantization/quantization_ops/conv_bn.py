import oneflow as flow
from quantization_ops.q_module import QModule, QParam

__all__ = ["QConvBN"]

class QConvBN(QModule):

    def __init__(self, conv_module, bn_module, qi=True, qo=True, quantization_bit=8, quantization_scheme='symmetric', quantization_formula='google', per_layer_quantization=True):
        super(QConvBN, self).__init__(qi=qi, qo=qo, quantization_bit=quantization_bit, quantization_scheme=quantization_scheme,
                                      quantization_formula=quantization_formula, per_layer_quantization=per_layer_quantization)
        self.quantization_bit = quantization_bit
        self.quantization_scheme = quantization_scheme
        self.quantization_formula = quantization_formula
        self.per_layer_quantization = per_layer_quantization
        self.conv_module = conv_module
        self.bn_module = bn_module
        self.qw = QParam(quantization_bit=quantization_bit, quantization_scheme=quantization_scheme,
                         quantization_formula=quantization_formula, per_layer_quantization=per_layer_quantization)
        self.quantization = flow.nn.Quantization(
            quantization_bit=32, quantization_scheme="affine", quantization_formula="google")

    def fold_bn(self, mean, std):
        if self.bn_module.affine:
            gamma_ = self.bn_module.weight / std
            weight = self.conv_module.weight * gamma_.view(self.conv_module.out_channels, 1, 1, 1)
            if self.conv_module.bias is not None:
                bias = gamma_ * self.conv_module.bias - gamma_ * mean + self.bn_module.bias
            else:
                bias = self.bn_module.bias - gamma_ * mean
        else:
            gamma_ = 1 / std
            weight = self.conv_module.weight * gamma_
            if self.conv_module.bias is not None:
                bias = gamma_ * self.conv_module.bias - gamma_ * mean
            else:
                bias = -gamma_ * mean
            
        return weight, bias


    def forward(self, x):

        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = self.qi.fake_quantize_tensor(x)

        if self.training:
            y = flow.F.conv2d(x, self.conv_module.weight, self.conv_module.bias, 
                            stride=self.conv_module.stride,
                            padding=self.conv_module.padding,
                            dilation=self.conv_module.dilation,
                            groups=self.conv_module.groups)
            y = y.permute(1, 0, 2, 3) # NCHW -> CNHW
            y = y.view(self.conv_module.out_channels, -1) # CNHW -> C,NHW
            mean = y.mean(1).detach()
            var = y.var(1).detach()
            self.bn_module.running_mean = \
                self.bn_module.momentum * self.bn_module.running_mean + \
                (1 - self.bn_module.momentum) * mean
            self.bn_module.running_var = \
                self.bn_module.momentum * self.bn_module.running_var + \
                (1 - self.bn_module.momentum) * var
        else:
            mean = flow.Tensor(self.bn_module.running_mean)
            var = flow.Tensor(self.bn_module.running_var)

        std = flow.sqrt(var + self.bn_module.eps)

        weight, bias = self.fold_bn(mean, std)

        self.qw.update(weight.data)

        x = flow.F.conv2d(x, self.qw.fake_quantize_tensor(weight), bias, 
                stride=self.conv_module.stride,
                padding=self.conv_module.padding, dilation=self.conv_module.dilation, 
                groups=self.conv_module.groups)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = self.qo.fake_quantize_tensor(x)

        return x

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
        self.M = self.qw.scale.numpy() * self.qi.scale.numpy() / self.qo.scale.numpy()

        weight, bias = self.fold_bn(self.bn_module.running_mean, self.bn_module.running_var)
        self.conv_module.weight = flow.nn.Parameter(
            self.qw.quantize_tensor(weight) - self.qw.zero_point)

        self.conv_module.bias = flow.nn.Parameter(self.quantization(
            bias, self.qi.scale * self.qw.scale, flow.Tensor([0])))

