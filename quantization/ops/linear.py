import oneflow as flow
from ..q_module import QParam, QModule

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
            raise ValueError('qo has been provided in init  function.')
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

        x = flow.F.matmul(x, self.qw.fake_quantize_tensor(self.fc_module.weight)) + self.fc_module.bias
 
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
