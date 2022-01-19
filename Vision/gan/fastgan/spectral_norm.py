import oneflow as flow
import oneflow.nn as nn
from oneflow.nn.parameter import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

def mv(x1, x2):
    return flow.matmul(x1, x2.view(-1, 1)).view(-1)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()
    
    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        # u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        # v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u = Parameter(flow.randn(height), requires_grad=False)
        v = Parameter(flow.randn(width), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(mv(w.view(height,-1).data.transpose(1, 0), u.data))
            u.data = l2normalize(mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = flow.dot(u, mv(w.view(height, -1), v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)