import oneflow as flow
import oneflow.nn as nn


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, input):
        return input * flow.sigmoid(input)


class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        n = input.shape[0]
        c_out = input.shape[1] // 2
        w_new = input.shape[2] * 2
        return input.view(n, c_out, w_new)


class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualLayer, self).__init__()

        self.conv1d_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.InstanceNorm1d(num_features=out_channels, affine=True),
        )

        self.conv_layer_gates = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.InstanceNorm1d(num_features=out_channels, affine=True),
        )

        self.conv1d_out_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.InstanceNorm1d(num_features=in_channels, affine=True),
        )

    def forward(self, input):
        h1_norm = self.conv1d_layer(input)
        h1_gates_norm = self.conv_layer_gates(input)

        # GLU
        h1_glu = h1_norm * flow.sigmoid(h1_gates_norm)

        h2_norm = self.conv1d_out_layer(h1_glu)
        return input + h2_norm


class downSample_Generator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(downSample_Generator, self).__init__()

        self.convLayer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.InstanceNorm2d(num_features=out_channels, affine=True),
        )
        self.convLayer_gates = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.InstanceNorm2d(num_features=out_channels, affine=True),
        )

    def forward(self, input):
        return self.convLayer(input) * flow.sigmoid(self.convLayer_gates(input))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 2D Conv Layer
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=128,
            kernel_size=(5, 15),
            stride=(1, 1),
            padding=(2, 7),
        )

        self.conv1_gates = nn.Conv2d(
            in_channels=1,
            out_channels=128,
            kernel_size=(5, 15),
            stride=1,
            padding=(2, 7),
        )

        # 2D Downsample Layer
        self.downSample1 = downSample_Generator(
            in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2
        )

        self.downSample2 = downSample_Generator(
            in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=2
        )

        # 2D -> 1D Conv
        self.conv2dto1dLayer = nn.Sequential(
            nn.Conv1d(
                in_channels=2304, out_channels=256, kernel_size=1, stride=1, padding=0
            ),
            nn.InstanceNorm1d(num_features=256, affine=True),
        )

        # Residual Blocks
        self.residualLayer1 = ResidualLayer(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.residualLayer2 = ResidualLayer(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.residualLayer3 = ResidualLayer(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.residualLayer4 = ResidualLayer(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.residualLayer5 = ResidualLayer(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.residualLayer6 = ResidualLayer(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
        )

        # 1D -> 2D Conv
        self.conv1dto2dLayer = nn.Sequential(
            nn.Conv1d(
                in_channels=256, out_channels=2304, kernel_size=1, stride=1, padding=0
            ),
            nn.InstanceNorm1d(num_features=2304, affine=True),
        )

        # UpSample Layer
        self.upSample1 = self.upSample(
            in_channels=256, out_channels=1024, kernel_size=5, stride=1, padding=2
        )

        self.upSample2 = self.upSample(
            in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=2
        )

        self.lastConvLayer = nn.Conv2d(
            in_channels=128,
            out_channels=1,
            kernel_size=(5, 15),
            stride=(1, 1),
            padding=(2, 7),
        )

    def downSample(self, in_channels, out_channels, kernel_size, stride, padding):
        self.ConvLayer = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.InstanceNorm1d(num_features=out_channels, affine=True),
            GLU(),
        )

        return self.ConvLayer

    def upSample(self, in_channels, out_channels, kernel_size, stride, padding):
        self.convLayer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.PixelShuffle(upscale_factor=2),
            nn.InstanceNorm2d(num_features=out_channels // 4, affine=True),
            GLU(),
        )
        return self.convLayer

    def forward(self, input):
        input = input.unsqueeze(1)
        conv1 = self.conv1(input) * flow.sigmoid(self.conv1_gates(input))

        # DownloadSample
        downsample1 = self.downSample1(conv1)
        downsample2 = self.downSample2(downsample1)

        # 2D -> 1D
        # reshape
        reshape2dto1d = downsample2.view(downsample2.size(0), 2304, 1, -1)
        reshape2dto1d = reshape2dto1d.squeeze(2)
        conv2dto1d_layer = self.conv2dto1dLayer(reshape2dto1d)

        residual_layer_1 = self.residualLayer1(conv2dto1d_layer)
        residual_layer_2 = self.residualLayer2(residual_layer_1)
        residual_layer_3 = self.residualLayer3(residual_layer_2)
        residual_layer_4 = self.residualLayer4(residual_layer_3)
        residual_layer_5 = self.residualLayer5(residual_layer_4)
        residual_layer_6 = self.residualLayer6(residual_layer_5)

        # 1D -> 2D
        conv1dto2d_layer = self.conv1dto2dLayer(residual_layer_6)
        # reshape
        reshape1dto2d = conv1dto2d_layer.unsqueeze(2)
        reshape1dto2d = reshape1dto2d.view(reshape1dto2d.size(0), 256, 9, -1)

        # UpSample
        upsample_layer_1 = self.upSample1(reshape1dto2d)
        upsample_layer_2 = self.upSample2(upsample_layer_1)

        output = self.lastConvLayer(upsample_layer_2)
        output = output.squeeze(1)
        return output


# PatchGAN
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.convLayer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            GLU(),
        )

        # DownSample Layer
        self.downSample1 = self.downSample(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=1,
        )

        self.downSample2 = self.downSample(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            stride=[2, 2],
            padding=1,
        )

        self.downSample3 = self.downSample(
            in_channels=512,
            out_channels=1024,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=1,
        )

        self.downSample4 = self.downSample(
            in_channels=1024,
            out_channels=1024,
            kernel_size=[1, 5],
            stride=(1, 1),
            padding=(0, 2),
        )

        # Conv Layer
        self.outputConvLayer = nn.Sequential(
            nn.Conv2d(
                in_channels=1024,
                out_channels=1,
                kernel_size=(1, 3),
                stride=[1, 1],
                padding=[0, 1],
            )
        )

    def downSample(self, in_channels, out_channels, kernel_size, stride, padding):
        convLayer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.InstanceNorm2d(num_features=out_channels, affine=True),
            GLU(),
        )
        return convLayer

    def forward(self, input):
        # input has shape [batch_size, num_features, time]
        # discriminator requires shape [batchSize, 1, num_features, time]
        input = input.unsqueeze(1)
        conv_layer_1 = self.convLayer1(input)

        downsample1 = self.downSample1(conv_layer_1)
        downsample2 = self.downSample2(downsample1)
        downsample3 = self.downSample3(downsample2)

        output = flow.sigmoid(self.outputConvLayer(downsample3))
        return output
