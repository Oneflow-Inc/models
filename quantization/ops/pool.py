import oneflow as flow
from ..q_module import QParam, QModule

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

        (x, indexs) = flow.F.max_pool_2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        return x

    def quantize_inference(self, x):
        return flow.F.max_pool2d(x, self.kernel_size, self.stride, self.padding)