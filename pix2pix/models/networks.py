import oneflow.experimental as flow
import oneflow.experimental.nn as nn


class Generator(nn.Module):
    def __init__(self, out_channels=3):
        super().__init__()
        self.g_d1 = Downsample(3, 64, apply_batchnorm=False)
        self.g_d2 = Downsample(64, 128)
        self.g_d3 = Downsample(128, 256)
        self.g_d4 = Downsample(256, 512)
        self.g_d5 = Downsample(512, 512)
        self.g_d6 = Downsample(512, 512)
        self.g_d7 = Downsample(512, 512)
        self.g_d8 = Downsample(512, 512)
        self.g_u7 = Upsample(512, 512, apply_dropout=True)
        self.g_u6 = Upsample(1024, 512, apply_dropout=True)
        self.g_u5 = Upsample(1024, 512, apply_dropout=True)
        self.g_u4 = Upsample(1024, 512)
        self.g_u3 = Upsample(1024, 256)
        self.g_u2 = Upsample(512, 128)
        self.g_u1 = Upsample(256, 64)
        self.g_u0_deconv = nn.ConvTranspose2d(
            128, out_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()
        nn.init.normal_(self.g_u0_deconv.weight, 0., 0.02)

    def forward(self, x):
        # (n, 64, 128, 128)
        d1 = self.g_d1(x)
        # (n, 128, 64, 64)
        d2 = self.g_d2(d1)
        # (n, 256, 32, 32)
        d3 = self.g_d3(d2)
        # (n, 512, 16, 16)
        d4 = self.g_d4(d3)
        # (n, 512, 8, 8)
        d5 = self.g_d5(d4)
        # (n, 512, 4, 4)
        d6 = self.g_d6(d5)
        # (n, 512, 2, 2)
        d7 = self.g_d7(d6)
        # (n, 512, 1, 1)
        d8 = self.g_d8(d7)
        # (n, 1024, 2, 2)
        u7 = self.g_u7(d8)
        u7 = flow.cat([u7, d7], 1)
        # (n, 1024, 4, 4)
        u6 = self.g_u6(u7)
        u6 = flow.cat([u6, d6], 1)
        # (n, 1024, 8, 8)
        u5 = self.g_u5(u6)
        u5 = flow.cat([u5, d5], 1)
        # (n, 1024, 16, 16)
        u4 = self.g_u4(u5)
        u4 = flow.cat([u4, d4], 1)
        # (n, 512, 32, 32)
        u3 = self.g_u3(u4)
        u3 = flow.cat([u3, d3], 1)
        # (n, 256, 64, 64)
        u2 = self.g_u2(u3)
        u2 = flow.cat([u2, d2], 1)
        # (n, 128, 128, 128)
        u1 = self.g_u1(u2)
        u1 = flow.cat([u1, d1], 1)
        # (n, 3, 256, 256)
        u0 = self.g_u0_deconv(u1)
        u0 = self.tanh(u0)
        return u0


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.d_d1 = Downsample(6, 64, apply_batchnorm=False)
        self.d_d2 = Downsample(64, 128)
        self.d_d3 = Downsample(128, 256)
        self.d_d4 = Downsample(256, 512, stride=1)
        self.d_conv = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        nn.init.normal_(self.d_conv.weight, 0., 0.02)

    def forward(self, x):
        # (n, 6, 256, 256)
        # (n, 64, 128, 128)
        d1 = self.d_d1(x)
        # (n, 64, 64, 64)
        d2 = self.d_d2(d1)
        # (n, 256, 32, 32)
        d3 = self.d_d3(d2)
        # (n, 512, 31, 31)
        d4 = self.d_d4(d3)
        # (n, 1, 30, 30)
        conv = self.d_conv(d4)

        return conv


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, apply_batchnorm=True, stride=2, padding=1):
        super().__init__()

        self.downconv = nn.Conv2d(in_channels, out_channels, kernel_size=4,
                                  stride=stride, padding=padding, bias=False)
        downnorm = nn.BatchNorm2d(out_channels)
        downrelu = nn.LeakyReLU(0.2, True)
        if apply_batchnorm:
            model = [self.downconv, downnorm, downrelu]
        else:
            model = [self.downconv, downrelu]
        self.model = nn.Sequential(*model)
        self._init_params()

    def _init_params(self):
        nn.init.normal_(self.downconv.weight, 0., 0.02)

    def forward(self, x):
        y = self.model(x)
        return y


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, apply_dropout=False):
        super().__init__()

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4,
                                         stride=2, padding=1, bias=False)
        self.upnorm = nn.BatchNorm2d(out_channels)
        uprelu = nn.ReLU(True)
        if apply_dropout:
            model = [self.upconv, self.upnorm, nn.Dropout(0.5), uprelu]
        else:
            model = [self.upconv, self.upnorm, uprelu]
        self.model = nn.Sequential(*model)
        self._init_params()

    def _init_params(self):
        nn.init.normal_(self.upconv.weight, 0., 0.02)

    def forward(self, x):
        y = self.model(x)
        return y
