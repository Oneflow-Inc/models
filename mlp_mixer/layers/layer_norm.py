import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
import numpy as np

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(flow.Tensor(flow.ones(features, dtype=flow.float32)))
        self.bias = nn.Parameter(flow.Tensor(flow.zeros(features, dtype=flow.float32)))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)

        std = x.std(dim=-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias