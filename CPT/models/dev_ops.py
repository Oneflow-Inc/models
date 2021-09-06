import oneflow as flow


class LayerNorm(flow.nn.Module):
    def __init__(self, features, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.weight = flow.nn.Parameter(
            flow.Tensor(flow.ones(features, dtype=flow.float32))
        )
        self.bias = flow.nn.Parameter(
            flow.Tensor(flow.zeros(features, dtype=flow.float32))
        )

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        return self.weight * (x - mean) / flow.sqrt(var + self.eps) + self.bias
