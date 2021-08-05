import oneflow as flow
import oneflow.nn as nn

__all__ = ["QParam", "QModule"]

class QParam:

    def __init__(self, quantization_bit=8, quantization_scheme='symmetric', quantization_formula='google', per_layer_quantization=True):
        self.quantization_bit = quantization_bit
        self.quantization_scheme = quantization_scheme
        self.quantization_formula = quantization_formula
        self.per_layer_quantization = per_layer_quantization
        self.scale = None
        self.zero_point = None
        self.min_max_observer = flow.nn.MinMaxObserver(quantization_formula=quantization_formula, quantization_bit=quantization_bit, quantization_scheme=quantization_scheme, per_layer_quantization=per_layer_quantization)
        self.fake_quantization = flow.nn.FakeQuantization(quantization_formula=quantization_formula, quantization_bit=quantization_bit, quantization_scheme=quantization_scheme)
        self.quantization = flow.nn.Quantization(quantization_formula=quantization_formula, quantization_bit=quantization_bit, quantization_scheme=quantization_scheme)

    def update(self, tensor):
        self.scale, self.zero_point = self.min_max_observer(tensor)

    def quantize_tensor(self, tensor):
        return self.quantization(tensor, self.scale, self.zero_point)
    
    def fake_quantize_tensor(self, tensor):
        return self.fake_quantization(tensor, self.scale, self.zero_point)

    def __str__(self):
        info = 'scale: %.10f ' % self.scale.numpy()
        info += 'zp: %d ' % self.zero_point.numpy()
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




