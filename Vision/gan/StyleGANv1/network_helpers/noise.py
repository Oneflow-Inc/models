import oneflow as flow
import oneflow.nn as nn


class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(flow.zeros(channels))

    def forward(self, x, noise):
        if noise is None:
            noise = flow.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1) * noise.to(x.device)