import oneflow as flow
from quantization_ops.q_module import QModule, QParam

__all__ = ["QLinear"]


class QLinear(QModule):

    def __init__(self, fc_module, qi=True, qo=True, quantization_bit=8, quantization_scheme='symmetric', quantization_formula='google', per_layer_quantization=True):
        super(QLinear, self).__init__(qi=qi, qo=qo, quantization_bit=quantization_bit,
                                      quantization_scheme='symmetric', quantization_formula='google', per_layer_quantization=True)
        self.quantization_bit = quantization_bit
        self.fc_module = fc_module
        self.fake_quantization = flow.nn.FakeQuantization(
            quantization_formula=quantization_formula, quantization_bit=quantization_bit, quantization_scheme=quantization_scheme)
        self.qw = QParam(quantization_bit=quantization_bit, quantization_scheme='symmetric',
                         quantization_formula='google', per_layer_quantization=True)
        self.quantization = flow.nn.Quantization(
            quantization_bit=32, quantization_scheme="affine", quantization_formula="google")

    def freeze(self, qi=None, qo=None):

        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init  function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        self.M = self.qw.scale.numpy() * self.qi.scale.numpy() / self.qo.scale.numpy()

        self.fc_module.weight = flow.nn.Parameter(
            self.qw.quantize_tensor(self.conv_module.weight) - self.qw.zero_point)
        self.fc_module.bias = flow.nn.Parameter(
            self.qw.quantize_tensor(self.conv_module.weight) - self.qw.zero_point)

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = self.qi.fake_quantize_tensor(x)

        self.qw.update(self.fc_module.weight)
        x = flow.F.matmul(x, self.qw.fake_quantize_tensor(
            self.fc_module.weight), transpose_a=False, transpose_b=True) + self.fc_module.bias

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = self.qo.fake_quantize_tensor(x)

        return x
