import oneflow as flow
import oneflow.nn as nn
from oneflow.nn.modules.pooling import AdaptiveAvgPool2d

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x):
        return self.fn(x) + x

class ConvMixer(nn.Module):
    def __init__(self, hidden_dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
        super().__init__()
        self.conv_mixer = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim),
            *[nn.Sequential(
                ResidualAdd(nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, groups=hidden_dim, padding=kernel_size // 2),
                    nn.GELU(),
                    nn.BatchNorm2d(hidden_dim)
                )),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(hidden_dim)
            ) for i in range(depth)],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x):
        x = self.conv_mixer(x)
        return x


def build_model(args):
    if args.model == "convmixer_1536_20":
        return ConvMixer(hidden_dim=1536, depth=20, patch_size=7, kernel_size=9)
    elif args.model == "convmixer_768_32":
        return ConvMixer(hidden_dim=768, depth=32, patch_size=7, kernel_size=7)
    elif args.model == "convmixer_1024_20":
        return ConvMixer(hidden_dim=1024, depth=20, patch_size=14, kernel_size=9)
    else:
        raise NotImplementedError("We only support convmixer_1536_20, convmixer_768_32, convmixer_1024_20 now")


